/*      This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <condition_variable>
#include <libethash/ethash.h>
#include <libdevcore/Worker.h>
#include <libdevcore/Common.h>
#include <libdevcore/SHA3.h>
#include "Exceptions.h"


namespace dev
{
namespace eth
{

struct Result {
	h256 value;
	h256 mixHash;
};

class EthashAux
{
public:
	struct LightAllocation {
		LightAllocation(h256 const& _seedHash);
		~LightAllocation();
		bytesConstRef data() const;
		Result compute(h256 const& _headerHash, uint64_t _nonce) const;
		ethash_light_t light;
		uint64_t size;
	};

	using LightType = std::shared_ptr<LightAllocation>;

	static h256 seedHash(unsigned _number);
	static uint64_t number(h256 const& _seedHash);

	static LightType light(h256 const& _seedHash);

	static Result eval(h256 const& _seedHash, h256 const& _headerHash, uint64_t  _nonce) noexcept;

private:
	EthashAux() = default;
	static EthashAux& get();

	mutable std::mutex x_lights;
	std::unordered_map<h256, LightType> m_lights;

	mutable std::mutex x_epochs;
	std::unordered_map<h256, unsigned> m_epochs;
	h256s m_seedHashes;
};

struct WorkPackage {
	WorkPackage() = default;

	void reset()
	{
		header = h256();
	}
	explicit operator bool() const
	{
		return header != h256();
	}

	h256 boundary;
	h256 header;	///< When h256() means "pause until notified a new work package is available".
	h256 seed;
	h256 job;

	uint64_t startNonce = 0;
	int exSizeBits = -1;
	int job_len = 8;
};

struct Solution {
	uint64_t nonce;
	h256 mixHash;
	WorkPackage work;
	bool stale;
};

}
}
