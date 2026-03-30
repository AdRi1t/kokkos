// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

struct TimesTwoTag {};

struct Functor {
  Kokkos::View<double> v;
  KOKKOS_FUNCTION void operator()() const { v() *= 3; }
  KOKKOS_FUNCTION void operator()(const TimesTwoTag) const { v() *= 2; }
};

struct TenTag {};

struct ReductionFunctor {
  KOKKOS_FUNCTION void operator()(int& res) const { res = 5; }
  KOKKOS_FUNCTION void operator()(const TenTag, int& res) const { res = 10; }
};

struct CombinedReductionFunctor {
  KOKKOS_FUNCTION void operator()(int& res1, int& res2) const {
    res1 = 5;
    res2 = 5;
  }
  KOKKOS_FUNCTION void operator()(const TenTag, int& res1, int& res2) const {
    res1 = 10;
    res2 = 10;
  }

  KOKKOS_FUNCTION void operator()(int& res1, int& res2, int& res3) const {
    res1 = 5;
    res2 = 5;
    res3 = 5;
  }
  KOKKOS_FUNCTION void operator()(const TenTag, int& res1, int& res2,
                                  int& res3) const {
    res1 = 10;
    res2 = 10;
    res3 = 10;
  }
};

void test() {
  // ParallelFor based API
  Kokkos::View<double> v("v");
  auto mirror = Kokkos::create_mirror_view(v);
  mirror()    = 5;
  Kokkos::deep_copy(v, mirror);

  Functor times_2_or_3{v};

  double expected = 5;

  // Minimal
  Kokkos::single(times_2_or_3);
  Kokkos::deep_copy(mirror, v);
  expected *= 3;
  EXPECT_EQ(expected, mirror());

  // Minimal lambda
  Kokkos::single(KOKKOS_LAMBDA() { v() += 2; });
  Kokkos::deep_copy(mirror, v);
  expected += 2;
  EXPECT_EQ(expected, mirror());

  // +kernal_name +WorkTag +ExecSpace
  Kokkos::single(
      "single+worktag+exec_space",
      Kokkos::SinglePolicy<TimesTwoTag, Kokkos::DefaultExecutionSpace>(),
      times_2_or_3);
  Kokkos::deep_copy(mirror, v);
  expected *= 2;
  EXPECT_EQ(expected, mirror());

  // +kernel_name
  Kokkos::single("single", times_2_or_3);
  Kokkos::deep_copy(mirror, v);
  expected *= 3;
  EXPECT_EQ(expected, mirror());

  // +WorkTag
  Kokkos::single(Kokkos::SinglePolicy<TimesTwoTag>(), times_2_or_3);
  Kokkos::deep_copy(mirror, v);
  expected *= 2;
  EXPECT_EQ(expected, mirror());

  // +WorkTag +kernel_name
  Kokkos::single("single+worktag", Kokkos::SinglePolicy<TimesTwoTag>(),
                 times_2_or_3);
  Kokkos::deep_copy(mirror, v);
  expected *= 2;
  EXPECT_EQ(expected, mirror());

  // +WorkTag +ExecSpace
  Kokkos::single(
      Kokkos::SinglePolicy<TimesTwoTag, Kokkos::DefaultExecutionSpace>(),
      times_2_or_3);
  Kokkos::deep_copy(mirror, v);
  expected *= 2;
  EXPECT_EQ(expected, mirror());

  // +ExecSpace
  Kokkos::single(Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(),
                 times_2_or_3);
  Kokkos::deep_copy(mirror, v);
  expected *= 3;
  EXPECT_EQ(expected, mirror());

  // +Policy +kernel_name
  Kokkos::single("single+policy",
                 Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(),
                 times_2_or_3);
  Kokkos::deep_copy(mirror, v);
  expected *= 3;
  EXPECT_EQ(expected, mirror());

  // +ExecSpace instance
  Kokkos::DefaultExecutionSpace exec_space;
  Kokkos::single(Kokkos::SinglePolicy(exec_space), times_2_or_3);
  exec_space.fence();
  Kokkos::deep_copy(mirror, v);
  expected *= 3;
  EXPECT_EQ(expected, mirror());
}

void test_one_ouput() {
  // ParallelReduce based API
  int val;
  ReductionFunctor f;

  // Full signature
  // Functor
  Kokkos::single("single with output+worktag+exec_space",
                 Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace, TenTag>(),
                 f, val);
  EXPECT_EQ(val, 10);

  // Lambda
  Kokkos::single(
      "single with output+lambda+exec_space",
      Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(),
      KOKKOS_LAMBDA(int& ret) { ret = 5; }, val);
  EXPECT_EQ(val, 5);

  // Minimal
  Kokkos::single(f, val);
  EXPECT_EQ(val, 5);

  // +kernel_name
  Kokkos::single("single with output", f, val);
  EXPECT_EQ(val, 5);

  // +Policy
  Kokkos::single(Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(), f, val);
  EXPECT_EQ(val, 5);

  // +kernel_name +Policy
  Kokkos::single("single with output+exec_space",
                 Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(), f, val);
  EXPECT_EQ(val, 5);

  // +Worktag
  Kokkos::single(Kokkos::SinglePolicy<TenTag>(), f, val);
  EXPECT_EQ(val, 10);

  // +Worktag +Policy
  Kokkos::single(Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace, TenTag>(),
                 f, val);
  EXPECT_EQ(val, 10);

  // +kernel_name +Worktag
  Kokkos::single("single with output+worktag", Kokkos::SinglePolicy<TenTag>(),
                 f, val);
  EXPECT_EQ(val, 10);

  // +ExecSpace instance
  Kokkos::DefaultExecutionSpace exec_space;
  Kokkos::single(Kokkos::SinglePolicy(exec_space), f, val);
  exec_space.fence();
  EXPECT_EQ(val, 5);
}

void test_multiple_outputs() {
  {
    int val1, val2;

    auto l = KOKKOS_LAMBDA(int& s1, int& s2) {
      s1 = 1;
      s2 = 2;
    };

    // Lambda
    // Minimal
    Kokkos::single(l, val1, val2);
    EXPECT_EQ(val1, 1);
    EXPECT_EQ(val2, 2);

    // +Label
    Kokkos::single("single with multiple outputs+lambda", l, val1, val2);
    EXPECT_EQ(val1, 1);
    EXPECT_EQ(val2, 2);

    // +Policy
    Kokkos::single(Kokkos::SinglePolicy(), l, val1, val2);
    EXPECT_EQ(val1, 1);
    EXPECT_EQ(val2, 2);

    // Full
    Kokkos::single("single with 2 outputs+lambda+policy",
                   Kokkos::SinglePolicy(), l, val1, val2);
    EXPECT_EQ(val1, 1);
    EXPECT_EQ(val2, 2);

    // Full with ExecSpace
    Kokkos::single("single with 2 outputs+lambda+exec_space",
                   Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(), l,
                   val1, val2);
    EXPECT_EQ(val1, 1);
    EXPECT_EQ(val2, 2);

    // Functor
    CombinedReductionFunctor f{};
    // Minimal
    Kokkos::single(f, val1, val2);
    EXPECT_EQ(val1, 5);
    EXPECT_EQ(val2, 5);

    // +Label
    Kokkos::single("single with 2 outputs", f, val1, val2);
    EXPECT_EQ(val1, 5);
    EXPECT_EQ(val2, 5);

    // +Policy
    Kokkos::single(Kokkos::SinglePolicy(), f, val1, val2);
    EXPECT_EQ(val1, 5);
    EXPECT_EQ(val2, 5);

    // +Policy with WorkTag
    Kokkos::single(Kokkos::SinglePolicy<TenTag>(), f, val1, val2);
    EXPECT_EQ(val1, 10);
    EXPECT_EQ(val2, 10);

    // Full
    Kokkos::single("single with 2 outputs+policy", Kokkos::SinglePolicy(), f,
                   val1, val2);
    EXPECT_EQ(val1, 5);
    EXPECT_EQ(val2, 5);

    // Full with WorkTag
    Kokkos::single("single with 2 outputs+worktag",
                   Kokkos::SinglePolicy<TenTag>(), f, val1, val2);
    EXPECT_EQ(val1, 10);
    EXPECT_EQ(val2, 10);

    // Full with WorkTag and ExecSpace
    Kokkos::single(
        "single with 2 outputs+worktag+execspace",
        Kokkos::SinglePolicy<TenTag, Kokkos::DefaultExecutionSpace>(), f, val1,
        val2);
    EXPECT_EQ(val1, 10);
    EXPECT_EQ(val2, 10);

    // With ExecSpace instance
    Kokkos::DefaultExecutionSpace exec_space;
    Kokkos::single(Kokkos::SinglePolicy(exec_space), f, val1, val2);
    exec_space.fence();
    EXPECT_EQ(val1, 5);
    EXPECT_EQ(val2, 5);
  }

  {
    int val1, val2, val3;

    auto l = KOKKOS_LAMBDA(int& s1, int& s2, int& s3) {
      s1 = 1;
      s2 = 2;
      s3 = 3;
    };

    // Lambda
    // Minimal
    Kokkos::single(l, val1, val2, val3);
    EXPECT_EQ(val1, 1);
    EXPECT_EQ(val2, 2);
    EXPECT_EQ(val3, 3);

    // +Label
    Kokkos::single("single with 3 outputs+lambda", l, val1, val2, val3);
    EXPECT_EQ(val1, 1);
    EXPECT_EQ(val2, 2);
    EXPECT_EQ(val3, 3);

    // +Policy
    Kokkos::single(Kokkos::SinglePolicy(), l, val1, val2, val3);
    EXPECT_EQ(val1, 1);
    EXPECT_EQ(val2, 2);
    EXPECT_EQ(val3, 3);

    // Full
    Kokkos::single("single with 3 outputs+lambda+policy",
                   Kokkos::SinglePolicy(), l, val1, val2, val3);
    EXPECT_EQ(val1, 1);
    EXPECT_EQ(val2, 2);
    EXPECT_EQ(val3, 3);

    // Full with ExecSpace
    Kokkos::single("single with 3 outputs+lambda+exec_space",
                   Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(), l,
                   val1, val2, val3);
    EXPECT_EQ(val1, 1);
    EXPECT_EQ(val2, 2);
    EXPECT_EQ(val3, 3);

    //// Functor
    CombinedReductionFunctor f{};
    // Minimal
    Kokkos::single(f, val1, val2, val3);
    EXPECT_EQ(val1, 5);
    EXPECT_EQ(val2, 5);
    EXPECT_EQ(val3, 5);

    // +Label
    Kokkos::single("single with 3 outputs", f, val1, val2, val3);
    EXPECT_EQ(val1, 5);
    EXPECT_EQ(val2, 5);
    EXPECT_EQ(val3, 5);

    // +Policy
    Kokkos::single(Kokkos::SinglePolicy(), f, val1, val2, val3);
    EXPECT_EQ(val1, 5);
    EXPECT_EQ(val2, 5);
    EXPECT_EQ(val3, 5);

    // +Policy with WorkTag
    Kokkos::single(Kokkos::SinglePolicy<TenTag>(), f, val1, val2, val3);
    EXPECT_EQ(val1, 10);
    EXPECT_EQ(val2, 10);
    EXPECT_EQ(val3, 10);

    // Full
    Kokkos::single("single with 3 outputs+policy", Kokkos::SinglePolicy(), f,
                   val1, val2, val3);
    EXPECT_EQ(val1, 5);
    EXPECT_EQ(val2, 5);
    EXPECT_EQ(val3, 5);

    // Full with WorkTag
    Kokkos::single("single with 3 outputs+worktag",
                   Kokkos::SinglePolicy<TenTag>(), f, val1, val2, val3);
    EXPECT_EQ(val1, 10);
    EXPECT_EQ(val2, 10);
    EXPECT_EQ(val3, 10);

    // Full with WorkTag and ExecSpace
    Kokkos::single(
        "single with 3 outputs+worktag+exec_space",
        Kokkos::SinglePolicy<TenTag, Kokkos::DefaultExecutionSpace>(), f, val1,
        val2, val3);
    EXPECT_EQ(val1, 10);
    EXPECT_EQ(val2, 10);
    EXPECT_EQ(val3, 10);

    // With ExecSpace instance
    Kokkos::DefaultExecutionSpace exec_space;
    Kokkos::single(Kokkos::SinglePolicy(exec_space), f, val1, val2, val3);
    exec_space.fence();
    EXPECT_EQ(val1, 5);
    EXPECT_EQ(val2, 5);
    EXPECT_EQ(val3, 5);
  }
}

namespace Test {
TEST(TEST_CATEGORY, single) { test(); }
TEST(TEST_CATEGORY, single_with_output) { test_one_ouput(); }
TEST(TEST_CATEGORY, single_with_multiple_outputs) { test_multiple_outputs(); }
}  // namespace Test
