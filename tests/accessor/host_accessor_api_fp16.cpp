/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::host_accessor api test for the sycl::half type
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable when sycl::accessor is implemented
#if !defined(__HIPSYCL__) && !defined(__COMPUTECPP__) && \
    !defined(__SYCL_COMPILER_VERSION)
#include "accessor_common.h"
#include "host_accessor_api_common.h"
#endif

namespace host_accessor_api_fp16 {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("sycl::host_accessor api. fp16 type", "[accessor]")({
  using namespace host_accessor_api_common;

  auto queue = sycl_cts::util::get_cts_object::queue();
  if (queue.get_device().has(sycl::aspect::fp16)) {
#ifndef SYCL_CTS_ENABLE_FULL_CONFORMANCE
    run_host_accessor_api_for_type<sycl::half>{}("sycl::half");
#else
    for_type_vectors_marray<run_host_accessor_api_for_type, sycl::half>(
        "sycl::half");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
  }
});
}  // namespace host_accessor_api_fp16
