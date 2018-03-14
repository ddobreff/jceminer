/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#include "CommonData.h"
#include <cstdlib>

using namespace std;
using namespace dev;

std::string dev::escaped(std::string const& _s, bool _all)
{
    static const map<char, char> prettyEscapes{{'\r', 'r'}, {'\n', 'n'}, {'\t', 't'}, {'\v', 'v'}};
    std::string ret;
    ret.reserve(_s.size() + 2);
    ret.push_back('"');
    for (auto i : _s)
        if (i == '"' && !_all)
            ret += "\\\"";
        else if (i == '\\' && !_all)
            ret += "\\\\";
        else if (prettyEscapes.count(i) && !_all) {
            ret += '\\';
            ret += prettyEscapes.find(i)->second;
        }
        else if (i < ' ' || _all) {
            ret += "\\x";
            ret.push_back("0123456789abcdef"[(uint8_t)i / 16]);
            ret.push_back("0123456789abcdef"[(uint8_t)i % 16]);
        }
        else
            ret.push_back(i);
    ret.push_back('"');
    return ret;
}

int dev::fromHex(char _i, WhenError _throw)
{
    if (_i >= '0' && _i <= '9')
        return _i - '0';
    if (_i >= 'a' && _i <= 'f')
        return _i - 'a' + 10;
    if (_i >= 'A' && _i <= 'F')
        return _i - 'A' + 10;
    if (_throw == WhenError::Throw)
        throw std::runtime_error("Bad hex character");
    else
        return -1;
}

bytes dev::fromHex(std::string const& _s, WhenError _throw)
{
    unsigned s = (_s[0] == '0' && _s[1] == 'x') ? 2 : 0;
    std::vector<uint8_t> ret;
    ret.reserve((_s.size() - s + 1) / 2);

    if (_s.size() % 2) {
        int h = fromHex(_s[s++], WhenError::DontThrow);
        if (h != -1)
            ret.push_back(h);
        else if (_throw == WhenError::Throw)
            throw std::runtime_error("Bad hex character");
        else
            return bytes();
    }
    for (unsigned i = s; i < _s.size(); i += 2) {
        int h = fromHex(_s[i], WhenError::DontThrow);
        int l = fromHex(_s[i + 1], WhenError::DontThrow);
        if (h != -1 && l != -1)
            ret.push_back((byte)(h * 16 + l));
        else if (_throw == WhenError::Throw)
            throw std::runtime_error("Bad hex character");
        else
            return bytes();
    }
    return ret;
}

bool dev::setenv(const char name[], const char value[], bool override)
{
    return ::setenv(name, value, override ? 1 : 0) == 0;
}
