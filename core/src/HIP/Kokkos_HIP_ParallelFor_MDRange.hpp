// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_HIP_PARALLEL_FOR_MDRANGE_HPP
#define KOKKOS_HIP_PARALLEL_FOR_MDRANGE_HPP

#include <Kokkos_Parallel.hpp>

#include <HIP/Kokkos_HIP_BlockSize_Deduction.hpp>
#include <HIP/Kokkos_HIP_KernelLaunch.hpp>
#include <KokkosExp_MDRangePolicy.hpp>
#include <impl/KokkosExp_IterateTileGPU.hpp>

namespace Kokkos {
namespace Impl {

// ParallelFor
template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>, HIP> {
 public:
  using Policy       = Kokkos::MDRangePolicy<Traits...>;
  using functor_type = FunctorType;

 private:
  using array_index_type = typename Policy::array_index_type;
  using index_type       = typename Policy::index_type;
  using LaunchBounds     = typename Policy::launch_bounds;
  using MaxGridSize      = Kokkos::Array<index_type, 3>;

  const FunctorType m_functor;
  const Policy m_policy;
  const MaxGridSize m_max_grid_size;

  array_type m_begins;
  array_type m_ends;

 public:
  ParallelFor()                              = delete;
  ParallelFor(ParallelFor const&)            = default;
  ParallelFor& operator=(ParallelFor const&) = delete;

  inline __device__ void operator()() const {
    Kokkos::Impl::DeviceIterate<Policy::rank, array_index_type, index_type,
                                FunctorType, Policy::inner_direction,
                                typename Policy::work_tag>(m_begins, m_ends,
                                                           m_functor)
        .exec_range();
  }

  inline void execute() const {
    using ClosureType = ParallelFor<FunctorType, Policy, HIP>;
    if (m_policy.m_num_tiles == 0) return;

    dim3 grid(1, 1, 1);
    dim3 block(1, 1, 1);
    if constexpr (RP::rank == 2) {
      const array_index_type block_0 = m_policy.m_tile[0];
      const array_index_type block_1 = m_policy.m_tile[1];

      const array_index_type grid_0 =
          (m_ends[0] - m_begins[0] + block_0 - 1) / block_0;
      const array_index_type grid_1 =
          (m_ends[1] - m_begins[1] + block_1 - 1) / block_1;

      if constexpr (RP::inner_direction == Iterate::Left) {
        block = dim3(block_0, block_1, 1);
        grid  = dim3(std::min<array_index_type>(grid_0, m_max_grid_size[0]),
                     std::min<array_index_type>(grid_1, m_max_grid_size[1]), 1);
      } else {
        block = dim3(block_1, block_0, 1);
        grid  = dim3(std::min<array_index_type>(grid_1, m_max_grid_size[0]),
                     std::min<array_index_type>(grid_0, m_max_grid_size[1]), 1);
      }
    } else if constexpr (RP::rank >= 3) {
      array_index_type block_0 = 1;
      array_index_type block_1 = 1;
      array_index_type block_2 = 1;

      if constexpr (RP::inner_direction == Iterate::Left) {
        if constexpr (RP::rank == 3) {
          block_0 = m_policy.m_tile[0];
          block_1 = m_policy.m_tile[1];
          block_2 = m_policy.m_tile[2];
        } else if constexpr (RP::rank >= 4) {
          block_0 = m_policy.m_tile[0] * m_policy.m_tile[1];
          block_1 = m_policy.m_tile[2];
          block_2 = m_policy.m_tile[3];
        }
        if constexpr (RP::rank >= 5) {
          block_1 = m_policy.m_tile[2] * m_policy.m_tile[3];
          block_2 = m_policy.m_tile[4];
        }
        if constexpr (RP::rank >= 6) {
          block_2 = m_policy.m_tile[4] * m_policy.m_tile[5];
        }
      } else {
        if constexpr (RP::rank == 3) {
          block_0 = m_policy.m_tile[2];
          block_1 = m_policy.m_tile[1];
          block_2 = m_policy.m_tile[0];
        } else if constexpr (RP::rank >= 4) {
          block_0 =
              m_policy.m_tile[RP::rank - 1] * m_policy.m_tile[RP::rank - 2];
          block_1 = m_policy.m_tile[RP::rank - 3];
          block_2 = m_policy.m_tile[RP::rank - 4];
        }
        if constexpr (RP::rank >= 5) {
          block_1 =
              m_policy.m_tile[RP::rank - 3] * m_policy.m_tile[RP::rank - 4];
          block_2 = m_policy.m_tile[RP::rank - 5];
        }
        if constexpr (RP::rank >= 6) {
          block_2 =
              m_policy.m_tile[RP::rank - 5] * m_policy.m_tile[RP::rank - 6];
        }
      }
      const array_index_type grid_0 =
          (m_ends[0] - m_begins[0] + block_0 - 1) / block_0;
      const array_index_type grid_1 =
          (m_ends[1] - m_begins[1] + block_1 - 1) / block_1;
      const array_index_type grid_2 =
          (m_ends[2] - m_begins[2] + block_2 - 1) / block_2;

      block = dim3(block_0, block_1, block_2);
      grid  = dim3(std::min<array_index_type>(grid_0, m_max_grid_size[0]),
                   std::min<array_index_type>(grid_1, m_max_grid_size[1]),
                   std::min<array_index_type>(grid_2, m_max_grid_size[2]));
    }

    hip_parallel_launch<ClosureType, LaunchBounds>(
        *this, grid, block, 0, m_policy.space().impl_internal_space_instance(),
        false);

  }  // end execute

  ParallelFor(FunctorType const& arg_functor, Policy const& arg_policy)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_max_grid_size({
            static_cast<index_type>(
                m_policy.space().hip_device_prop().maxGridSize[0]),
            static_cast<index_type>(
                m_policy.space().hip_device_prop().maxGridSize[1]),
            static_cast<index_type>(
                m_policy.space().hip_device_prop().maxGridSize[2]),
        }) {
    // Initialize begins and ends based on layout
    // Swap the fastest indexes to x dimension
    for (array_index_type i = 0; i < Policy::rank; ++i) {
      if constexpr (RP::inner_direction == Iterate::Left) {
        m_begins[i] = m_policy.m_lower[i];
        m_ends[i]   = m_policy.m_upper[i];
      } else {
        m_begins[i] = m_policy.m_lower[Policy::rank - 1 - i];
        m_ends[i]   = m_policy.m_upper[Policy::rank - 1 - i];
      }
    }
  }

  template <typename Policy, typename Functor>
  static int max_tile_size_product(const Policy&, const Functor&) {
    using closure_type =
        ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>, HIP>;
    unsigned block_size = hip_get_max_blocksize<closure_type, LaunchBounds>();
    if (block_size == 0)
      Kokkos::Impl::throw_runtime_exception(
          std::string("Kokkos::Impl::ParallelFor< HIP > could not find a valid "
                      "tile size."));
    return block_size;
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif
