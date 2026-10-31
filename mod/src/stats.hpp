#pragma once
#include <Geode/Geode.hpp>

// Per-level record of what the AI has done, kept in the mod's save data.
struct LevelStats {
    int64_t id = 0;
    std::string name;
    int stars = 0;
    bool demon = false;
    int demonDifficulty = 0;
    int attempts = 0;
    float bestPercent = 0.f;
    bool completed = false;
    int attemptsToComplete = 0;

    // Higher = harder. Demons always rank above star levels.
    int difficultyScore() const;
    std::string difficultyName() const;
};

namespace stats {
    std::vector<LevelStats>& all();
    LevelStats& forLevel(GJGameLevel* level);
    int totalAttempts();
    LevelStats const* hardestCompleted();
    void save();
}
