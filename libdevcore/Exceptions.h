/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#pragma once

#include <exception>
#include <string>
#include <boost/exception/all.hpp>
#include <boost/throw_exception.hpp>
#include "CommonData.h"
#include "FixedHash.h"

namespace dev
{

/// Base class for all exceptions.
struct Exception: virtual std::exception, virtual boost::exception {
	Exception(std::string _message = std::string()): m_message(std::move(_message)) {}
	const char* what() const noexcept override
	{
		return m_message.empty() ? std::exception::what() : m_message.c_str();
	}

private:
	std::string m_message;
};

#define DEV_SIMPLE_EXCEPTION(X) struct X: virtual Exception { const char* what() const noexcept override { return #X; } }

DEV_SIMPLE_EXCEPTION(BadHexCharacter);
DEV_SIMPLE_EXCEPTION(WorkerFailure);

// error information to be added to exceptions
//using errinfo_invalidSymbol = boost::error_info<struct tag_invalidSymbol, char>;
//using errinfo_comment = boost::error_info<struct tag_comment, std::string>;
//using errinfo_required = boost::error_info<struct tag_required, bigint>;
//using errinfo_got = boost::error_info<struct tag_got, bigint>;
//using RequirementError = boost::tuple<errinfo_required, errinfo_got>;

}
