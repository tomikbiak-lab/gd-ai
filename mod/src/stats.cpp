#include "stats.hpp"

using namespace geode::prelude;

namespace {
    std::vector<LevelStats> s_levels;
    bool s_loaded = false;

    void load() {
        s_loaded = true;
        auto saved = Mod::get()->getSavedValue<matjson::Value>("levels", matjson::Value::array());
        for (auto const& v : saved) {
            LevelStats s;
            s.id = v["id"].asInt().unwrapOr(0);
            s.name = v["name"].asString().unwrapOr("?");
            s.stars = static_cast<int>(v["stars"].asInt().unwrapOr(0));
            s.demon = v["demon"].asBool().unwrapOr(false);
            s.demonDifficulty = static_cast<int>(v["demonDifficulty"].asInt().unwrapOr(0));
            s.attempts = static_cast<int>(v["attempts"].asInt().unwrapOr(0));
            s.bestPercent = static_cast<float>(v["bestPercent"].asDouble().unwrapOr(0.0));
            s.completed = v["completed"].asBool().unwrapOr(false);
            s.attemptsToComplete = static_cast<int>(v["attemptsToComplete"].asInt().unwrapOr(0));
            s_levels.push_back(std::move(s));
        }
    }

    // GD stores demon difficulty as 0 = hard, 3 = easy, 4 = medium, 5 = insane, 6 = extreme.
    int demonRank(int demonDifficulty) {
        switch (demonDifficulty) {
            case 3: return 1;
            case 4: return 2;
            case 5: return 4;
            case 6: return 5;
            default: return 3;
        }
    }
}

int LevelStats::difficultyScore() const {
    return demon ? 100 + demonRank(demonDifficulty) : stars;
}

std::string LevelStats::difficultyName() const {
    if (demon) {
        constexpr char const* names[] = {"", "Easy", "Medium", "Hard", "Insane", "Extreme"};
        return fmt::format("{} Demon", names[demonRank(demonDifficulty)]);
    }
    if (stars <= 0) return "Unrated";
    if (stars == 1) return "Auto/Easy 1*";
    if (stars == 2) return "Easy 2*";
    if (stars == 3) return "Normal 3*";
    if (stars <= 5) return fmt::format("Hard {}*", stars);
    if (stars <= 7) return fmt::format("Harder {}*", stars);
    return fmt::format("Insane {}*", stars);
}

std::vector<LevelStats>& stats::all() {
    if (!s_loaded) load();
    return s_levels;
}

LevelStats& stats::forLevel(GJGameLevel* level) {
    auto& levels = all();
    int64_t id = level->m_levelID.value();
    std::string name = level->m_levelName;
    // Unsaved editor levels all have id 0, so tell them apart by name.
    auto it = std::find_if(levels.begin(), levels.end(), [&](LevelStats const& s) {
        return s.id == id && (id != 0 || s.name == name);
    });
    if (it == levels.end()) {
        levels.push_back({});
        it = levels.end() - 1;
        it->id = id;
    }
    // Refresh in case the level got rated or renamed since.
    it->name = name;
    it->stars = level->m_stars.value();
    it->demon = level->m_demon.value() != 0;
    it->demonDifficulty = level->m_demonDifficulty;
    return *it;
}

int stats::totalAttempts() {
    int total = 0;
    for (auto const& s : all()) total += s.attempts;
    return total;
}

LevelStats const* stats::hardestCompleted() {
    LevelStats const* hardest = nullptr;
    for (auto const& s : all()) {
        if (s.completed && (!hardest || s.difficultyScore() > hardest->difficultyScore())) hardest = &s;
    }
    return hardest;
}

void stats::save() {
    auto arr = matjson::Value::array();
    for (auto const& s : all()) {
        auto v = matjson::Value::object();
        v["id"] = s.id;
        v["name"] = s.name;
        v["stars"] = s.stars;
        v["demon"] = s.demon;
        v["demonDifficulty"] = s.demonDifficulty;
        v["attempts"] = s.attempts;
        v["bestPercent"] = s.bestPercent;
        v["completed"] = s.completed;
        v["attemptsToComplete"] = s.attemptsToComplete;
        arr.push(v);
    }
    Mod::get()->setSavedValue("levels", arr);
}
