/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

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

    mutable std::mutex x_work;              ///< Lock for the network existance.
    std::unique_ptr<std::thread> m_work;        ///< The network thread.
    std::atomic<WorkerState> m_state = {WorkerState::Starting};
};

}
