// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_CUDA_MDRANGEPOLICY_HPP_
#define KOKKOS_CUDA_MDRANGEPOLICY_HPP_

#include <KokkosExp_MDRangePolicy.hpp>

namespace Kokkos {

template <>
struct default_outer_direction<Kokkos::Cuda> {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Left;
};

template <>
struct default_inner_direction<Kokkos::Cuda> {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Left;
};

namespace Impl {

template <>
struct TileSizeRecommended<Kokkos::Cuda> {
  template <typename Policy>
  static auto get(Policy const&) {
    constexpr auto InnerDirection = Policy::inner_direction;
    constexpr int Rank            = Policy::rank;

    using tile_type = typename Policy::tile_type;

    tile_type tile_sizes{};
    if constexpr (Rank == 2) {
      tile_sizes = tile_type{64, 4};
    } else if constexpr (Rank == 3) {
      tile_sizes = tile_type{32, 2, 4};
    } else if constexpr (Rank == 4) {
      tile_sizes = tile_type{16, 2, 2, 4};
    } else if constexpr (Rank == 5) {
      tile_sizes = tile_type{16, 2, 4, 2, 1};
    } else if constexpr (Rank == 6) {
      tile_sizes = tile_type{8, 4, 2, 2, 2, 1};
    } else {
      for (int i = 0; i < Rank; ++i) {
        tile_sizes[i] = 2;
      }
      tile_sizes[0] = 16;
    }

    if constexpr (InnerDirection == Iterate::Left) {
      return tile_sizes;
    } else {
      // Reverse the tile sizes for right inner direction
      tile_type reversed_tile_sizes{};
      for (int i = 0; i < Rank; ++i) {
        reversed_tile_sizes[i] = tile_sizes[Rank - 1 - i];
      }
      return reversed_tile_sizes;
    }
  }

  template <typename Policy>
  static auto get(Policy const&, const int max_tile_size) {
    constexpr auto InnerDirection = Policy::inner_direction;
    constexpr int Rank            = Policy::rank;
    constexpr int default_inner_tile = (Rank < 4) ? 32 : 16;
    constexpr int default_tile = (Rank < 3) ? 4 : 2;

    using tile_type = typename Policy::tile_type;

    int inner_tile = std::min(default_inner_tile, max_tile_size);

    tile_type tile_sizes{};
    int prod_tile_dims = inner_tile;
    tile_sizes[0] = inner_tile;

    for (int i = 1; i < Rank; ++i) {
      if (prod_tile_dims * default_tile <= max_tile_size) {
        tile_sizes[i] = default_tile;
      } else {
        tile_sizes[i] = 1;
      } 
      prod_tile_dims *= tile_sizes[i];
    }

    if constexpr (InnerDirection == Iterate::Left) {
      return tile_sizes;
    } else {
      // Reverse the tile sizes for right inner direction
      tile_type reversed_tile_sizes{};
      for (int i = 0; i < Rank; ++i) {
        reversed_tile_sizes[i] = tile_sizes[Rank - 1 - i];
      }
      return reversed_tile_sizes;
    }
  }
};

// Settings for MDRangePolicy
template <>
inline TileSizeProperties get_tile_size_properties<Kokkos::Cuda>(
    const Kokkos::Cuda& space) {
  TileSizeProperties properties;
  const auto& device_prop        = space.cuda_device_prop();
  properties.max_threads         = device_prop.maxThreadsPerMultiProcessor;
  properties.max_total_tile_size = 512;
  properties.max_threads_dimensions[0] = device_prop.maxThreadsDim[0];
  properties.max_threads_dimensions[1] = device_prop.maxThreadsDim[1];
  properties.max_threads_dimensions[2] = device_prop.maxThreadsDim[2];
  return properties;
}

// Settings for TeamMDRangePolicy
template <typename Rank, TeamMDRangeThreadAndVector ThreadAndVector>
struct ThreadAndVectorNestLevel<Rank, Cuda, ThreadAndVector>
    : AcceleratorBasedNestLevel<Rank, ThreadAndVector> {};

}  // Namespace Impl
}  // Namespace Kokkos
#endif
