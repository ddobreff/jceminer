/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#include <pthread.h>
#include <signal.h>
#include <boost/stacktrace.hpp>
#include "Common.h"
#include "Worker.h"
#include "Log.h"

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
			m_state.compare_exchange_strong(ex, WorkerState::Started);
			try {
				workLoop();
			} catch (std::exception const& _e) {
				logerror("Exception thrown in " << workerName() << ": " << _e.what());
				abort();
			}

			ex = m_state.exchange(WorkerState::Stopped);
			logerror(workerName() << " unexpectedly stopped");
			abort();
		}));
	}
	while (m_state == WorkerState::Starting)
		this_thread::sleep_for(chrono::microseconds(20));
}

Worker::~Worker()
{
}
