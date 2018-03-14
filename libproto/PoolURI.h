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

#include <network/uri.hpp>
#include <string>

// A simple URI parser specifically for mining pool enpoints
namespace dev
{

enum class SecureLevel {NONE = 0, TLS12, TLS};

class URI : network::uri
{
public:
	URI();
	URI(const std::string uri);

	std::string	Scheme() const;
	std::string	Host() const;
	unsigned short	Port() const;
	std::string	User() const;
	std::string	Pswd() const;
	SecureLevel	ProtoSecureLevel() const;
	unsigned	ProtoVersion() const;

	bool		KnownScheme();

	static std::string KnownSchemes(SecureLevel secureLevel);

private:
	network::uri   m_uri;
};

}
