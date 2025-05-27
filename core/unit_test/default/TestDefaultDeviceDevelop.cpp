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

  KOKKOS_FUNCTION void operator()(const TimeTwo) const { v(0) *= 2; }

  KOKKOS_FUNCTION void operator()(const PlusOne) const { ++v(0); }
};

void test_func() {
  Kokkos::View<double*> v("v", 1);
  auto mirror = Kokkos::create_mirror_view(v);
  mirror(0)   = 5;
  Kokkos::deep_copy(v, mirror);

  auto l = KOKKOS_LAMBDA() { v(0) *= 2; };

  Funct f;
  f.v = v;

  double res = 5;

  // Full signature
  Kokkos::single<Kokkos::DefaultExecutionSpace, TimeTwo>("Single", f);
  Kokkos::deep_copy(mirror, v);
  res *= 2;
  EXPECT_EQ(res, mirror(0));
  Kokkos::single<Kokkos::DefaultExecutionSpace, void>("Single", l);
  Kokkos::deep_copy(mirror, v);
  res *= 2;
  EXPECT_EQ(res, mirror(0));

  // Minimal
  Kokkos::single(l);
  Kokkos::deep_copy(mirror, v);
  res *= 2;
  EXPECT_EQ(res, mirror(0));

  // +kernel_name
  Kokkos::single("test", l);
  Kokkos::deep_copy(mirror, v);
  res *= 2;
  EXPECT_EQ(res, mirror(0));

  // +WorkTag
  Kokkos::single<PlusOne>(f);
  Kokkos::deep_copy(mirror, v);
  res += 1;
  EXPECT_EQ(res, mirror(0));

  // +WorkTag +kernel_name
  Kokkos::single<TimeTwo>("test", f);
  Kokkos::deep_copy(mirror, v);
  res *= 2;
  EXPECT_EQ(res, mirror(0));

  // +WorkTag +ExecSpace
  Kokkos::single<TimeTwo, Kokkos::DefaultExecutionSpace>(f);
  Kokkos::deep_copy(mirror, v);
  res *= 2;
  EXPECT_EQ(res, mirror(0));

  // +WorkTag +ExecSpace +kernel_name
  Kokkos::single<TimeTwo, Kokkos::DefaultExecutionSpace>("test", f);
  Kokkos::deep_copy(mirror, v);
  res *= 2;
  EXPECT_EQ(res, mirror(0));

  // +ExecSpace
  Kokkos::single<Kokkos::DefaultExecutionSpace>(l);
  Kokkos::deep_copy(mirror, v);
  res *= 2;
  EXPECT_EQ(res, mirror(0));

  // +ExecSpace +kernel_name
  Kokkos::single<Kokkos::DefaultExecutionSpace>("test", l);
  Kokkos::deep_copy(mirror, v);
  res *= 2;
  EXPECT_EQ(res, mirror(0));

  // +ExecSpace +WorkTag
  Kokkos::single<Kokkos::DefaultExecutionSpace, TimeTwo>(f);
  Kokkos::deep_copy(mirror, v);
  res *= 2;
  EXPECT_EQ(res, mirror(0));

  // +ExecSpace +WorkTag +kernel_name
  Kokkos::single<Kokkos::DefaultExecutionSpace, TimeTwo>("test", f);
  Kokkos::deep_copy(mirror, v);
  res *= 2;
  EXPECT_EQ(res, mirror(0));
}

namespace Test {

TEST(defaultdevicetype, development_test) { test_func(); }

}  // namespace Test
