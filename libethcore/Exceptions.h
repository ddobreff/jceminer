// This source code is licenced under GNU General Public License, Version 3.

#pragma once

#include <libdevcore/Exceptions.h>

namespace dev
{
namespace eth
{

// information to add to exceptions
using errinfo_name = boost::error_info<struct tag_field, std::string>;
using errinfo_field = boost::error_info<struct tag_field, int>;
using errinfo_data = boost::error_info<struct tag_data, std::string>;
using errinfo_nonce = boost::error_info<struct tag_nonce, h64>;
using errinfo_difficulty = boost::error_info<struct tag_difficulty, u256>;
using errinfo_target = boost::error_info<struct tag_target, h256>;
using errinfo_seedHash = boost::error_info<struct tag_seedHash, h256>;
using errinfo_mixHash = boost::error_info<struct tag_mixHash, h256>;
using errinfo_ethashResult = boost::error_info<struct tag_ethashResult, std::tuple<h256, h256>>;
using BadFieldError = boost::tuple<errinfo_field, errinfo_data>;

DEV_SIMPLE_EXCEPTION(DAGCreationFailure);
DEV_SIMPLE_EXCEPTION(LiteCreationFailure);
DEV_SIMPLE_EXCEPTION(DAGComputeFailure);
DEV_SIMPLE_EXCEPTION(StratumTimeout);
DEV_SIMPLE_EXCEPTION(GPUFailure);
DEV_SIMPLE_EXCEPTION(SignalFailure);

}
}
