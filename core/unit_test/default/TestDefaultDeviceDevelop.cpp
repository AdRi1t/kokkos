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

namespace Test {

template <int Rank, typename ScalarType, typename Layout>
struct ViewTypeRank {
  using type = void;
};

template <typename ScalarType, typename Layout>
struct ViewTypeRank<1, ScalarType, Layout> {
  using type = Kokkos::View<ScalarType *, Layout>;
};

template <typename ScalarType, typename Layout>
struct ViewTypeRank<2, ScalarType, Layout> {
  using type = Kokkos::View<ScalarType **, Layout>;
};

template <typename ScalarType, typename Layout>
struct ViewTypeRank<3, ScalarType, Layout> {
  using type = Kokkos::View<ScalarType ***, Layout>;
};

template <typename ScalarType, typename Layout>
struct ViewTypeRank<4, ScalarType, Layout> {
  using type = Kokkos::View<ScalarType ****, Layout>;
};

template <typename ScalarType, typename Layout>
struct ViewTypeRank<5, ScalarType, Layout> {
  using type = Kokkos::View<ScalarType *****, Layout>;
};

template <typename ScalarType, typename Layout>
struct ViewTypeRank<6, ScalarType, Layout> {
  using type = Kokkos::View<ScalarType ******, Layout>;
};

struct Bound_4 {
  static constexpr int value = 4;
};

struct Bound_8 {
  static constexpr int value = 8;
};

struct Bound_16 {
  static constexpr int value = 16;
};

struct Bound_32 {
  static constexpr int value = 32;
};

template <typename BoundTag, int Rank>
struct Functor_Heavy {
  using view_type =
      typename ViewTypeRank<Rank, double, Kokkos::LayoutLeft>::type;

  static constexpr int bound = BoundTag::value;

  Functor_Heavy(int n, const view_type &view) : m_N(n), m_view_a(view) {}

  template <typename... Idxs>
  KOKKOS_INLINE_FUNCTION void operator()(BoundTag, int i0, Idxs... idx) const {
    double intermediates[bound];
    constexpr int warpSize = 32;

    for (int i = 0; i < bound; ++i) {
      intermediates[i] = m_N * i0 % 256;
    }

    double x = m_view_a(i0, idx...);

    for (int i = 0; i < bound; ++i) {
      x += Kokkos::sin(Kokkos::pow(
          Kokkos::atan2(x, i % warpSize) * intermediates[(i - 1) % bound],
          intermediates[i % bound]));

      if (i0 % 4 == 0) {
        x += Kokkos::sqrt(static_cast<double>(i + i0) *
                          intermediates[(i * 2) % bound] * 0.01);
      } else if (i0 % 4 == 1) {
        x *= Kokkos::sin(static_cast<double>(i0 * i) *
                         intermediates[(i * 3) % bound] * 0.01);
      } else if (i0 % 4 == 2) {
        x /= (Kokkos::cos(
            static_cast<double>(i / (i0 + 1) * intermediates[(i - 1) % bound] *
                                0.01) +
            1.1));
      } else {
        x -= 1 / (i + 1 * Kokkos::tan(intermediates[i0 % bound] +
                                      static_cast<double>(i - i0) * i0));
      }

      intermediates[i % bound] = x;
    }

    for (int i = 0; i < bound; ++i) {
      intermediates[i] =
          intermediates[i] / intermediates[(i0 + warpSize - 1) % bound];
      intermediates[i] =
          intermediates[(i0 - warpSize - 1) % bound] / intermediates[i];
    }

    if (x == m_N) {
      m_view_a(i0, idx...) = x;
    } else {
      m_view_a(i0, idx...) = -x;
    }
  }

  int m_N;
  view_type m_view_a;
};

TEST(defaultdevicetype, development_test) {
  // Kokkos::View<double **, Kokkos::LayoutLeft> view_2D("2D_view", 128, 128);
  Kokkos::View<double ***, Kokkos::LayoutLeft> view_3D("3D_view", 128, 128,
                                                       128);
  Kokkos::parallel_for("3D exec",
                       Kokkos::MDRangePolicy<Kokkos::Rank<3>, Bound_4>(
                           {0, 0, 0}, {128, 128, 128}),
                       Functor_Heavy<Bound_4, 3>(1024, view_3D));
  Kokkos::fence("Fence after exec");

  Kokkos::parallel_for("3D exec",
                       Kokkos::MDRangePolicy<Kokkos::Rank<3>, Bound_8>(
                           {0, 0, 0}, {128, 128, 128}),
                       Functor_Heavy<Bound_8, 3>(1024, view_3D));
  Kokkos::fence("Fence after exec");

  Kokkos::parallel_for("3D exec",
                       Kokkos::MDRangePolicy<Kokkos::Rank<3>, Bound_16>(
                           {0, 0, 0}, {128, 128, 128}),
                       Functor_Heavy<Bound_16, 3>(1024, view_3D));
  Kokkos::fence("Fence after exec");

  Kokkos::parallel_for("3D exec",
                       Kokkos::MDRangePolicy<Kokkos::Rank<3>, Bound_32>(
                           {0, 0, 0}, {128, 128, 128}),
                       Functor_Heavy<Bound_32, 3>(1024, view_3D));
  Kokkos::fence("Fence after exec");
}

}  // namespace Test
