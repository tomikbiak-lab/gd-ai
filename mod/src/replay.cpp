#include "replay.hpp"

using namespace geode::prelude;

namespace {
    std::optional<int64_t> s_requested;

    std::filesystem::path dir() {
        auto path = Mod::get()->getSaveDir() / "replays";
        std::error_code ec;
        std::filesystem::create_directories(path, ec);
        return path;
    }

    std::filesystem::path pathFor(int64_t levelID) {
        return dir() / fmt::format("{}.json", levelID);
    }

    matjson::Value toJson(Replay const& r) {
        auto presses = matjson::Value::array();
        for (auto const& [tick, held] : r.presses) {
            auto press = matjson::Value::array();
            press.push(tick);
            press.push(held ? 1 : 0);
            presses.push(press);
        }
        auto v = matjson::Value::object();
        v["version"] = 1;
        v["levelID"] = r.levelID;
        v["levelName"] = r.levelName;
        v["percent"] = r.percent;
        v["won"] = r.won;
        v["ticks"] = r.ticks;
        v["presses"] = presses;
        return v;
    }
}

std::optional<Replay> replay::load(int64_t levelID) {
    auto text = file::readString(pathFor(levelID));
    if (!text) return std::nullopt;
    auto json = matjson::parse(*text);
    if (!json) return std::nullopt;
    auto& v = *json;
    Replay r;
    r.levelID = v["levelID"].asInt().unwrapOr(levelID);
    r.levelName = v["levelName"].asString().unwrapOr("");
    r.percent = static_cast<float>(v["percent"].asDouble().unwrapOr(0.0));
    r.won = v["won"].asBool().unwrapOr(false);
    r.ticks = static_cast<int>(v["ticks"].asInt().unwrapOr(0));
    for (auto const& press : v["presses"]) {
        r.presses.push_back({static_cast<int>(press[0].asInt().unwrapOr(0)), press[1].asInt().unwrapOr(0) != 0});
    }
    return r;
}

bool replay::exists(int64_t levelID) {
    std::error_code ec;
    return std::filesystem::exists(pathFor(levelID), ec);
}

bool replay::saveIfBest(Replay const& r) {
    if (auto old = load(r.levelID)) {
        // The first win is the one worth keeping; before that, only further runs replace the replay.
        if (old->won || (!r.won && old->percent >= r.percent)) return false;
    }
    auto res = file::writeString(pathFor(r.levelID), toJson(r).dump());
    if (!res) {
        log::warn("GD AI: couldn't save replay: {}", res.unwrapErr());
        return false;
    }
    if (r.won) {
        // Also kept under its own name, so a later reset of the best replay can't lose it.
        auto stamp = fmt::format("{}-win-{}.json", r.levelID, std::time(nullptr));
        (void)file::writeString(dir() / stamp, toJson(r).dump());
    }
    log::info("GD AI: saved replay for {} ({:.1f}%{})", r.levelName, r.percent, r.won ? ", win" : "");
    return true;
}

void replay::request(int64_t levelID) {
    s_requested = levelID;
}

std::optional<Replay> replay::takeRequest(int64_t levelID) {
    if (!s_requested || *s_requested != levelID) return std::nullopt;
    s_requested.reset();
    return load(levelID);
}
