// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <TestDefaultDeviceType_Category.hpp>

struct TimeTwo {};
struct PlusOne {};

struct Funct {
  Kokkos::View<double*> v;
  KOKKOS_FUNCTION void operator()() const { v(0) *= 3; }
  KOKKOS_FUNCTION void operator()(const TimeTwo) const { v(0) *= 2; }
  KOKKOS_FUNCTION void operator()(const PlusOne) const { ++v(0); }
};

void test_func() {
  Kokkos::DefaultExecutionSpace space;

  Kokkos::View<double*> v("v", 1);
  auto mirror = Kokkos::create_mirror_view(v);
  mirror(0)   = 5;
  Kokkos::deep_copy(v, mirror);

  Funct f;
  f.v = v;

  double res = 5;

  // Minimal
  Kokkos::single(f);
  Kokkos::deep_copy(mirror, v);
  res *= 3;
  EXPECT_EQ(res, mirror(0));

  // Minimal lambda
  Kokkos::single(KOKKOS_LAMBDA() { v(0) += 2; });
  Kokkos::deep_copy(mirror, v);
  res += 2;
  EXPECT_EQ(res, mirror(0));

  // +kernal_name +WorkTag +ExecSpace
  Kokkos::single("Single",
                 Kokkos::SinglePolicy<TimeTwo, Kokkos::DefaultExecutionSpace>(),
                 f);
  Kokkos::deep_copy(mirror, v);
  res *= 2;
  EXPECT_EQ(res, mirror(0));

  // +kernel_name
  Kokkos::single("test", f);
  Kokkos::deep_copy(mirror, v);
  res *= 3;
  EXPECT_EQ(res, mirror(0));

  // +WorkTag
  Kokkos::single(Kokkos::SinglePolicy<PlusOne>(), f);
  Kokkos::deep_copy(mirror, v);
  res += 1;
  EXPECT_EQ(res, mirror(0));

  // +WorkTag +kernel_name
  Kokkos::single("Single", Kokkos::SinglePolicy<TimeTwo>(), f);
  Kokkos::deep_copy(mirror, v);
  res *= 2;
  EXPECT_EQ(res, mirror(0));

  // +WorkTag +ExecSpace
  Kokkos::single(Kokkos::SinglePolicy<TimeTwo, Kokkos::DefaultExecutionSpace>(),
                 f);
  Kokkos::deep_copy(mirror, v);
  res *= 2;
  EXPECT_EQ(res, mirror(0));

  // +ExecSpace
  Kokkos::single(Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(), f);
  Kokkos::deep_copy(mirror, v);
  res *= 3;
  EXPECT_EQ(res, mirror(0));

  // +Policy +kernel_name
  Kokkos::single("Single",
                 Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(), f);
  Kokkos::deep_copy(mirror, v);
  res *= 3;
  EXPECT_EQ(res, mirror(0));
}

namespace Test {

TEST(defaultdevicetype, development_test) { test_func(); }

}  // namespace Test
