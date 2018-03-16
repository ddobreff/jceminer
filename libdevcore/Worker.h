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

#include <string>
#include <thread>
#include <atomic>
#include <cassert>
#include <mutex>

namespace dev
{

enum class WorkerState {
	Starting,
	Started,
	Stopped
};

class Worker
{
public:
	Worker(std::string const& _name): m_name(_name) {}

	Worker(Worker const&) = delete;
	Worker& operator=(Worker const&) = delete;

	virtual ~Worker();

	/// Starts worker thread; causes startedWorking() to be called.
	void startWorking();

	const std::string& workerName()
	{
		return m_name;
	}

private:
	virtual void workLoop() = 0;

	std::string m_name;

	mutable std::mutex x_work;				///< Lock for the network existance.
	std::unique_ptr<std::thread> m_work;		///< The network thread.
	std::atomic<WorkerState> m_state = {WorkerState::Starting};
};

}
