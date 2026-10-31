#pragma once
#include <Geode/Geode.hpp>

// A replay is just when the AI pressed and released, counted in 1/240 s ticks
// from the start of the level. Physics runs in exactly those ticks, so playing
// the same presses back reproduces the run.
struct Replay {
    int64_t levelID = 0;
    std::string levelName;
    float percent = 0.f;
    bool won = false;
    int ticks = 0;
    std::vector<std::pair<int, bool>> presses; // tick, held from that tick on
};

namespace replay {
    // Saves the replay as this level's best if it got further (or won).
    // Returns true when it was saved.
    bool saveIfBest(Replay const& replay);
    std::optional<Replay> load(int64_t levelID);
    bool exists(int64_t levelID);

    // Ask for the next PlayLayer of this level to play its replay instead of the AI.
    void request(int64_t levelID);
    std::optional<Replay> takeRequest(int64_t levelID);
}
