/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#pragma once

// way to many unsigned to size_t warnings in 32 bit build
#ifdef _M_IX86
#pragma warning(disable:4244)
#endif

#include <map>
#include <unordered_map>
#include <vector>
#include <set>
#include <unordered_set>
#include <functional>
#include <string>
#include <chrono>
#include <mutex>
#include <boost/functional/hash.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include "vector_ref.h"

using byte = uint8_t;

namespace dev
{


typedef std::lock_guard<std::mutex> Guard;

// Binary data types.
using bytes = std::vector<byte>;
using bytesRef = vector_ref<byte>;
using bytesConstRef = vector_ref<byte const>;

template <class T>
class secure_vector
{
public:
	secure_vector() {}
	secure_vector(secure_vector<T> const& _c) = default;
	explicit secure_vector(unsigned _size): m_data(_size) {}
	explicit secure_vector(unsigned _size, T _item): m_data(_size, _item) {}
	explicit secure_vector(std::vector<T> const& _c): m_data(_c) {}
	explicit secure_vector(vector_ref<T> _c): m_data(_c.data(), _c.data() + _c.size()) {}
	explicit secure_vector(vector_ref<const T> _c): m_data(_c.data(), _c.data() + _c.size()) {}
	~secure_vector()
	{
		ref().cleanse();
	}

	secure_vector<T>& operator=(secure_vector<T> const& _c)
	{
		if (&_c == this)
			return *this;

		ref().cleanse();
		m_data = _c.m_data;
		return *this;
	}
	std::vector<T>& writable()
	{
		clear();
		return m_data;
	}
	std::vector<T> const& makeInsecure() const
	{
		return m_data;
	}

	void clear()
	{
		ref().cleanse();
	}

	vector_ref<T> ref()
	{
		return vector_ref<T>(&m_data);
	}
	vector_ref<T const> ref() const
	{
		return vector_ref<T const>(&m_data);
	}

	size_t size() const
	{
		return m_data.size();
	}
	bool empty() const
	{
		return m_data.empty();
	}

	void swap(secure_vector<T>& io_other)
	{
		m_data.swap(io_other.m_data);
	}

private:
	std::vector<T> m_data;
};

using bytesSec = secure_vector<byte>;

// Numeric types.
using bigint = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<>>;
using u64 =
    boost::multiprecision::number<boost::multiprecision::cpp_int_backend<64, 64, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using u128 =
    boost::multiprecision::number<boost::multiprecision::cpp_int_backend<128, 128, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using u256 =
    boost::multiprecision::number<boost::multiprecision::cpp_int_backend<256, 256, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using s256 =
    boost::multiprecision::number<boost::multiprecision::cpp_int_backend<256, 256, boost::multiprecision::signed_magnitude, boost::multiprecision::unchecked, void>>;
using u160 =
    boost::multiprecision::number<boost::multiprecision::cpp_int_backend<160, 160, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using s160 =
    boost::multiprecision::number<boost::multiprecision::cpp_int_backend<160, 160, boost::multiprecision::signed_magnitude, boost::multiprecision::unchecked, void>>;
using u512 =
    boost::multiprecision::number<boost::multiprecision::cpp_int_backend<512, 512, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using u256s = std::vector<u256>;
using u160s = std::vector<u160>;
using u256Set = std::set<u256>;
using u160Set = std::set<u160>;

// String types.
using strings = std::vector<std::string>;

// Null/Invalid values for convenience.
static const u256 Invalid256 = ~(u256)0;

/// Converts arbitrary value to string representation using std::stringstream.
template <class _T>
std::string toString(_T const& _t)
{
	std::ostringstream o;
	o << _t;
	return o.str();
}

}
