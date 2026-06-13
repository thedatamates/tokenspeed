// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

// Coverage: block-diffusion retraction retention primitives — retract pins on
// snapshot nodes (vs admission pruning and device-eviction detach) and
// PagedCacheGroupTable::TrimTailOwned (canvas-span trim).

#include "paged_cache_test_fixture.h"

namespace tokenspeed::test {

using PagedCacheRetentionTest = PagedCacheSmallFixture;

// ------------------------------------------------------------
//  Block-diffusion retraction retention: retract pins keep snapshot-owned
//  pages alive (against both admission pruning and device-eviction detach)
//  while a retracted request's retained tables borrow them, and
//  TrimTailOwned returns exactly the discarded canvas span.
// ------------------------------------------------------------

// Admission pruning must skip retract-pinned snapshots; dropping the pin
// makes the same admission succeed by pruning them.
TEST_F(PagedCacheRetentionTest, RetractPinBlocksAdmissionPrune_UnpinAllows) {
    auto* leaf = InsertDevicePages(/*num_pages=*/2, /*token_start=*/1);  // depth 4 = kLcm
    ASSERT_NE(leaf, nullptr);
    hybrid_->AttachPagedCacheSnapshotToNode(leaf, MakeCompleteSnapshot(kLcm));
    ASSERT_TRUE(leaf->HasPagedCacheSnapshot());
    // Usable pool = total_pages - 1 (dummy page): 15 per group.
    const std::int32_t swa_free_after_attach = swa_alloc_->AvailablePages();
    ASSERT_LT(swa_free_after_attach, 15);

    // Simulate diffusion retraction of the request whose tables borrow from
    // this chain ("victim" has no tables here; only the pin matters).
    hybrid_->RetainRequestTablesForRetract("victim", leaf, kLcm);

    // "other" wants 30 raw tokens: swa (raw 2/page, 15 usable) only fits if
    // the snapshot's swa pages are pruned. Pinned -> admission must FAIL
    // without touching the snapshot.
    {
        auto simulated_free = hybrid_->InitialSimulatedFree();
        EXPECT_FALSE(hybrid_->AdmitChunk("other", 0, 30, simulated_free));
        EXPECT_TRUE(leaf->HasPagedCacheSnapshot());
        EXPECT_EQ(swa_alloc_->AvailablePages(), swa_free_after_attach);
    }

    // Unpinned (resume/finish dropped the retract pin) -> the same admission
    // succeeds by pruning the state portion; its pages return to the pool.
    hybrid_->ReleaseRetractPin("victim");
    {
        auto simulated_free = hybrid_->InitialSimulatedFree();
        EXPECT_TRUE(hybrid_->AdmitChunk("other", 0, 30, simulated_free));
        EXPECT_EQ(swa_alloc_->AvailablePages(), 15);
    }
}

// Device LRU eviction must leave a retract-pinned snapshot attached (the node
// survives the device drop because the retraction writeback left it
// host-resident); after the pin drops, the detach proceeds as usual.
TEST(PagedCacheRetractPinEviction, DeviceEvictionKeepsPinnedSnapshotAttached) {
    constexpr std::int32_t kPageSize = 2;
    PageAllocator device_alloc{kPageSize, /*total_pages=*/8};
    PageAllocator host_alloc{kPageSize, /*total_pages=*/8};
    KVPrefixCache kv{&device_alloc, &host_alloc};

    PagedCacheGroupConfig fh{};
    fh.group_id = "fh";
    fh.rows_per_page = 4;
    fh.entry_stride_tokens = 1;
    fh.total_pages = 16;
    fh.retention = PagedCacheGroupConfig::Retention::FullHistory;
    auto fh_owner = std::make_unique<PagedCacheGroupAllocator>(fh);
    auto* fh_alloc = fh_owner.get();

    HybridPrefixCache hybrid{kv, /*mamba=*/nullptr, /*mamba_chunk_size=*/0};
    hybrid.RegisterPagedCacheGroup(std::move(fh_owner));
    hybrid.EnablePagedCacheAdjunct({"fh"}, {});
    kv.GetDeviceManager().SetEvictionCallback([&hybrid](TreeNode* node) { hybrid.OnKVEvict(node); });

    // Branch A: 2 device pages (4 tokens = one fh page), then give the chain
    // host copies (as the retraction writeback would).
    auto tokens_a = MakeAlignedTokens(2, kPageSize, /*start=*/1);
    {
        OwnedPages pages = device_alloc.Allocate(2);
        kv.Insert<ResourceType::Device>(tokens_a, /*prefix_pages=*/{}, std::move(pages));
    }
    auto match_a = kv.Match(tokens_a);
    TreeNode* terminal = match_a.device.last_node;
    ASSERT_NE(terminal, nullptr);
    ASSERT_TRUE(kv.AllocateResourceOfType<ResourceType::Host>(match_a.NodesWithout<ResourceType::Host>()));
    HostNodeRef host_pin{terminal};  // Retracted holds the host chain

    // Snapshot holding one fh page, attached at the terminal; pin it.
    {
        PagedCacheGroupTable table{fh_alloc};
        table.Acquire(4);
        auto committed = table.CommitHistoryToSnapshot(4);
        auto snap = std::make_unique<PagedCacheSnapshot>();
        snap->prefix_len_tokens = 4;
        PagedCacheGroupSnapshot group_snap{};
        group_snap.pages = std::move(committed.pages);
        group_snap.raw_token_cursor = 4;
        snap->groups.emplace("fh", std::move(group_snap));
        ASSERT_TRUE(hybrid.AttachPagedCacheSnapshotToNode(terminal, std::move(snap)));
    }
    ASSERT_EQ(fh_alloc->AvailablePages(), 14);  // 15 usable - 1 snapshot-held
    hybrid.RetainRequestTablesForRetract("victim", terminal, 4);

    // Branch B pinned on device so the LRU must evict branch A.
    auto tokens_b = MakeAlignedTokens(2, kPageSize, /*start=*/100);
    {
        OwnedPages pages = device_alloc.Allocate(2);
        kv.Insert<ResourceType::Device>(tokens_b, /*prefix_pages=*/{}, std::move(pages));
    }
    DeviceNodeRef ref_b{kv.Match(tokens_b).device.last_node};

    const std::int32_t target_available = device_alloc.AvailablePages() + 1;
    ASSERT_TRUE(kv.EnsureCapacityByEvict<ResourceType::Device>(target_available));

    // Pinned: the snapshot survived the device eviction (the node is still
    // alive on host) and its group page was NOT freed.
    EXPECT_TRUE(terminal->HasPagedCacheSnapshot());
    EXPECT_EQ(fh_alloc->AvailablePages(), 14);

    // Pin dropped: a later eviction notification detaches as usual.
    hybrid.ReleaseRetractPin("victim");
    hybrid.OnKVEvict(terminal);
    EXPECT_FALSE(terminal->HasPagedCacheSnapshot());
    EXPECT_EQ(fh_alloc->AvailablePages(), 15);
}

// TrimTailOwned: drops exactly the owned tail beyond the keep boundary,
// rewinds the cursor, never touches borrowed/committed coverage.
TEST(PagedCacheGroupTableTrim, TrimTailOwnedDropsCanvasSpanOnly) {
    PagedCacheGroupConfig fh{};
    fh.group_id = "fh";
    fh.rows_per_page = 4;
    fh.entry_stride_tokens = 1;
    fh.total_pages = 16;
    fh.retention = PagedCacheGroupConfig::Retention::FullHistory;
    PagedCacheGroupAllocator alloc{fh};

    PagedCacheGroupTable table{&alloc};
    table.Acquire(22);  // 6 pages: covers committed 14 + canvas 8
    auto committed = table.CommitHistoryToSnapshot(12);  // 3 pages -> borrowed-mirrored
    ASSERT_EQ(table.Size(), 6);
    ASSERT_EQ(table.BorrowedPagesCount(), 3);
    ASSERT_EQ(table.OwnedPagesCount(), 3);

    // Retraction at committed 14: only the canvas span (2 pages) returns.
    auto released = table.TrimTailOwned(14);
    EXPECT_EQ(released.size(), 2u);
    EXPECT_EQ(table.Size(), 4);
    EXPECT_EQ(table.RawTokenCursor(), 14);
    EXPECT_EQ(table.CommittedPrefixLenTokens(), 12);
    EXPECT_TRUE(table.TrimTailOwned(14).empty());  // idempotent

    // Trimming below the committed prefix is a logic error.
    EXPECT_THROW(table.TrimTailOwned(8), std::logic_error);

    // Resume re-extends the same table over the fresh canvas span.
    table.Acquire(22);
    EXPECT_EQ(table.Size(), 6);
    EXPECT_EQ(table.RawTokenCursor(), 22);

    // Borrowed pages are never trimmed: keep=12 leaves exactly the mirrored
    // committed coverage.
    auto released2 = table.TrimTailOwned(12);
    EXPECT_EQ(released2.size(), 3u);
    EXPECT_EQ(table.Size(), 3);
    EXPECT_EQ(table.BorrowedPagesCount(), 3);
    EXPECT_EQ(table.OwnedPagesCount(), 0);
}

}  // namespace tokenspeed::test
