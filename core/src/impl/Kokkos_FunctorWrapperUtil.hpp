// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_KOKKOS_FUNCTORWRAPPERUTIL_HPP
#define KOKKOS_IMPL_KOKKOS_FUNCTORWRAPPERUTIL_HPP

#include <type_traits>

namespace Kokkos::Impl {

// Helper to allow passing an indexless functor to the parallel_for backend
// through special interface such as Kokkos::GraphNodeThen and Kokkos::Single.
template <typename Functor>
struct IndexlessFunctorWrapper {
  Functor m_functor;

  template <typename WorkTagOrIndex, typename... MaybeIndex>
  KOKKOS_FUNCTION void operator()(WorkTagOrIndex, MaybeIndex...) const {
    static_assert(sizeof...(MaybeIndex) <= 1);
    if constexpr (sizeof...(MaybeIndex) == 0) {
      m_functor();
    } else {
      static_assert(std::is_empty_v<WorkTagOrIndex>);
      m_functor(WorkTagOrIndex{});
    }
  }
};

// Helper to allow passing an indexless functor to the parallel_reduce backend
// through special interface such as Kokkos::GraphNodeThen and Kokkos::Single.
template <class FunctorType>
struct IndexlessReductionFunctorWrapper {
  FunctorType m_functor;

  template <typename WorkTagOrIndex, class IndexOrReturnType,
            typename... MaybeReturnType>
  KOKKOS_FUNCTION void operator()(const WorkTagOrIndex,
                                  IndexOrReturnType& indexOrRet,
                                  MaybeReturnType&... maybeRet) const {
    static_assert(sizeof...(MaybeReturnType) <= 1);
    if constexpr (sizeof...(MaybeReturnType) == 0) {
      m_functor(indexOrRet);
    } else {
      static_assert(std::is_empty_v<WorkTagOrIndex>);
      m_functor(WorkTagOrIndex{}, maybeRet...);
    }
  }
};

}  //  namespace Kokkos::Impl

#endif  // KOKKOS_IMPL_KOKKOS_FUNCTORWRAPPERUTIL_HPP
