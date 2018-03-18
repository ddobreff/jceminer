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

#include "Common.h"
#include "Worker.h"
#include "Log.h"
#include <iostream>
#include <chrono>
#include <thread>

using namespace std;
using namespace dev;

void Worker::startWorking()
{
	Guard l(x_work);
	if (m_work) {
		WorkerState ex = WorkerState::Stopped;
		m_state.compare_exchange_strong(ex, WorkerState::Starting);
	} else {
		m_state = WorkerState::Starting;
		m_work.reset(new thread([&]() {
			WorkerState ex = WorkerState::Starting;
			bool ok = m_state.compare_exchange_strong(ex, WorkerState::Started);
			(void)ok;

			try {
				workLoop();
			} catch (std::exception const& _e) {
				logerror << "Exception thrown in Worker thread: " << _e.what() << endl;
				exit(-1);
			}

			ex = m_state.exchange(WorkerState::Stopped);
			logerror << "Worker unexpectedly stopped\n";
			exit(-1);

		}));
	}
	while (m_state == WorkerState::Starting)
		this_thread::sleep_for(chrono::microseconds(20));
}

Worker::~Worker()
{
}
