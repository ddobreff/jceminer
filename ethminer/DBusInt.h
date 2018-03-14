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

#include <dbus/dbus.h>

using namespace std;

class DBusInt
{
public:
	DBusInt()
	{
		dbus_error_init(&err);
		conn = dbus_bus_get(DBUS_BUS_SESSION, &err);
		if (!conn)
			cerr << "DBus error " << err.name << ": " << err.message << endl;
		dbus_bus_request_name(conn, "eth.miner", DBUS_NAME_FLAG_REPLACE_EXISTING, &err);
		if (dbus_error_is_set(&err)) {
			cerr << "DBus error " << err.name << ": " << err.message << endl;
			dbus_connection_close(conn);
		}
		cout << "DBus initialized!" << endl;
	}

	void send(const char* hash)
	{
		DBusMessage* msg;
		msg = dbus_message_new_signal("/eth/miner/hash", "eth.miner.monitor", "Hash");
		if (msg == nullptr)
			cerr << "Message is null!" << endl;
		dbus_message_append_args(msg, DBUS_TYPE_STRING, &hash, DBUS_TYPE_INVALID);
		if (!dbus_connection_send(conn, msg, NULL)) cerr << "Error sending message!" << endl;
		dbus_message_unref(msg);
	}

private:
	DBusError err;
	DBusConnection* conn;
};
