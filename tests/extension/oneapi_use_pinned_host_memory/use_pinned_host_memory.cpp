/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Provides tests to check handler class functions exceptions with fp64 type
//  gained with oneapi_memcpy2d extension
//
*******************************************************************************/

#include "../../common/common.h"
#include "../../../util/sycl_exceptions.h"

using use_pinned_host_memory = typename
    sycl::ext::oneapi::property::buffer::use_pinned_host_memory;
static sycl::property_list pl {use_pinned_host_memory()};

template <typename AllocatorT, int dims>
void run_test_for_allocator() {
  using T = typename AllocatorT::value_type;
  auto range = sycl_cts::util::get_cts_object::range<dims>::get(1, 1, 1);

  sycl::buffer<T, dims> buf1(range, pl);
  sycl::buffer<T, dims, AllocatorT> buf2(range, AllocatorT{}, pl);

  CHECK(buf1.template has_property<use_pinned_host_memory>());

  auto get_prop = [&]() {
    auto prop = buf1.template get_property<use_pinned_host_memory>();
  };
  CHECK_NOTHROW(get_prop());
}

template <typename T, int dims>
void run_test_for_dim() {
  run_test_for_allocator<sycl::buffer_allocator<T>, dims>();
  run_test_for_allocator<std::allocator<T>, dims>();
}

template <typename T>
void run_test_for_type() {
   run_test_for_dim<T, 1>();
   run_test_for_dim<T, 2>();
   run_test_for_dim<T, 3>();
}

TEST_CASE("buffer constructors without host data", "[use_pinned_host_memory]") {
#if !defined(SYCL_EXT_ONEAPI_USE_PINNED_HOST_MEMORY_PROPERTY)
  SKIP("SYCL_EXT_ONEAPI_USE_PINNED_HOST_MEMORY_PROPERTY is not defined");
#else
  run_test_for_type<int>();
  run_test_for_type<float>();
#endif
}

template <typename T>
void run_test_for_host_data() {
  static constexpr auto buffer_size = 10;
  static const sycl::range<1> range(buffer_size);
  static const std::array<T, buffer_size> data{};

  auto t_ptr_constructor = [&]() {
    sycl::buffer<T> buf(data.data(), range, pl);
  };
  auto container_constructor = [&]() {
    sycl::buffer<T> buf(data, pl);
  };
  auto shared_ptr_constructor = [&]() {
    std::shared_ptr<T> data_shared(new T[buffer_size], std::default_delete<T[]>());
    sycl::buffer<T> buf(data_shared, range, pl);
  };
  auto shared_ptr_arr_constructor = [&]() {
    std::shared_ptr<T[]> data_shared(new T[buffer_size]);
    sycl::buffer<T> buf(data_shared, range, pl);
  };
  auto iterator_constructor = [&]() {
    sycl::buffer<T> buf(data.begin(), data.end(), pl);
  };

  CHECK_THROWS_MATCHES(
      t_ptr_constructor(), sycl::exception,
      sycl_cts::util::equals_exception(sycl::errc::invalid));

  CHECK_THROWS_MATCHES(
      container_constructor(), sycl::exception,
      sycl_cts::util::equals_exception(sycl::errc::invalid));

  CHECK_THROWS_MATCHES(
      shared_ptr_constructor(), sycl::exception,
      sycl_cts::util::equals_exception(sycl::errc::invalid));

  CHECK_THROWS_MATCHES(
      shared_ptr_arr_constructor(), sycl::exception,
      sycl_cts::util::equals_exception(sycl::errc::invalid));

  CHECK_THROWS_MATCHES(
      iterator_constructor(), sycl::exception,
      sycl_cts::util::equals_exception(sycl::errc::invalid));
}

TEST_CASE("buffer constructors with host data", "[use_pinned_host_memory]") {
#if !defined(SYCL_EXT_ONEAPI_USE_PINNED_HOST_MEMORY_PROPERTY)
  SKIP("SYCL_EXT_ONEAPI_USE_PINNED_HOST_MEMORY_PROPERTY is not defined");
#else
  run_test_for_host_data<int>();
  run_test_for_host_data<float>();
#endif
}
