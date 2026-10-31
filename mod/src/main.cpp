#include <Geode/Geode.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include "levels.hpp"
#include "net.hpp"
#include "replay.hpp"
#include "stats.hpp"

using namespace geode::prelude;

// Protocol v4, spoken with brain/brain2.py. What the AI "sees": grids of cells
// with one channel per kind of thing, plus numbers about the players. Three
// grids: around player 1, a coarser "far" one around player 1 that reaches
// twice as far ahead, and around player 2 in dual mode. The same layout works
// on every level, which is what lets what it learns carry over to the next.
// New grids and numbers are only ever appended, so older brains still fit.
constexpr int32_t PROTOCOL = 4;
constexpr int COLS = 16; // 1 behind the player, 15 ahead
constexpr int ROWS = 11;
constexpr float CELL = 30.f;
constexpr float FAR_CELL = 60.f;
enum Kind {
    HAZARD, SOLID, JUMP_PAD, GRAVITY_PAD, JUMP_ORB, GRAVITY_ORB,
    GRAVITY_PORTAL, MODE_PORTAL, OTHER_PORTAL, SPEED_PORTAL, KINDS
};
enum Grid { NEAR_P1 = 0, NEAR_P2 = KINDS, FAR_P1 = 2 * KINDS, CHANNELS = 3 * KINDS };
constexpr int GRID = CHANNELS * ROWS * COLS;
// Numbers: 0-15 player 1, 16-31 dual mode / player 2, 32 ticks since the last press or release,
// 33-67 the 5 nearest things ahead with exact distance, size, kind and movement (7 each).
constexpr int NEAREST = 5;
constexpr int NEAREST_VALUES = 7;
constexpr int EXTRA = 33 + NEAREST * NEAREST_VALUES;
constexpr float TICK = 1.f / 240.f;
// The brain decides once per window of this many ticks, and says on which tick
// of the window to press or release, so clicks still land on the exact tick.
constexpr int DECISION_TICKS = 4;
// Practice: a checkpoint every this many px. Ones placed in the last second
// before a death are thrown away, since that state may already be doomed.
constexpr float CHECKPOINT_SPACING = 90.f;
constexpr int DOOMED_TICKS = 240;
// At high game speed, one frame can run many ticks. Past this much real time the rest waits for
// the next frame, so the picture stays smooth (the game just runs a bit slower when the PC can't keep up).
constexpr auto FRAME_BUDGET = std::chrono::milliseconds(12);

namespace {
    struct Look {
        int channel;
        uint8_t value;
    };

    struct CachedObject {
        GameObject* obj;
        Look look;
    };

    struct CheckpointMeta {
        int attempt;
        int tick;
    };

    // Little-endian message builder for the brain.
    struct Packet {
        std::vector<uint8_t> data;
        explicit Packet(char tag) { data.push_back(static_cast<uint8_t>(tag)); }
        template <class T>
        Packet& put(T const& value) {
            auto bytes = reinterpret_cast<uint8_t const*>(&value);
            data.insert(data.end(), bytes, bytes + sizeof(T));
            return *this;
        }
        Packet& putBytes(void const* bytes, size_t len) {
            auto p = static_cast<uint8_t const*>(bytes);
            data.insert(data.end(), p, p + len);
            return *this;
        }
        bool send() const { return net::sendAll(data.data(), data.size()); }
    };

    std::vector<CachedObject> s_objects; // sorted by x
    bool s_cacheBuilt = false;
    bool s_held = false;
    bool s_needPlan = true;     // ask the brain how to start the next attempt
    bool s_applyingPlan = false;
    bool s_pendingReset = false;
    bool s_dying = false;       // death animation playing, GD will restart by itself
    bool s_won = false;
    bool s_inAIStep = false;
    bool s_removeFirstArg = false; // which removeCheckpoint() flag pops the last one
    int s_steps = 0;
    int s_windowTick = 0;      // tick within the current decision window
    bool s_planHold = false;   // what the brain wants held from s_planTick on
    int s_planTick = 0;
    bool s_windowStartHeld = false;
    int s_attempt = 0;
    int s_doomedAttempt = -1;   // attempt whose last checkpoints must be dropped
    int s_deathTick = 0;
    int s_practiceStartsLogged = 0;
    float s_practiceStart = 0.f;    // where the coach wants practice to start; checkpoints only go up to it
    int s_player2DeathsLogged = 0;
    float s_tickTime = 0.f;     // real time not yet turned into ticks
    uint8_t s_grid[GRID];
    float s_extra[EXTRA];
    int s_ticksSinceSwitch = 0;                                   // since the last press or release
    std::unordered_map<GameObject*, CCPoint> s_lastSeen;          // where nearby objects were last decision
    std::unordered_map<CheckpointObject*, CheckpointMeta> s_checkpointMeta;
    // Replays: presses of the current full run from the start, and a replay being played back.
    std::vector<std::pair<int, bool>> s_presses;
    bool s_recordingRun = false;
    std::optional<Replay> s_replay;
    size_t s_replayIndex = 0;
    bool s_replayFresh = false;
    bool s_cursorShown = false;
    bool s_switching = false;   // brain asked for another level; waiting for it to open
    bool s_quietWin = false;    // finishing the level just starts the next attempt (no end screen) // GD hides the cursor in levels; shown again while the AI plays
    CCDrawNode* s_visionNode = nullptr;

    Look lookFor(GameObject* obj) {
        if (obj->m_isPassable || obj->m_isNoTouch) return {-1, 0};
        // Speed portals are told apart by object ID, not type.
        switch (obj->m_objectID) {
            case 200: return {SPEED_PORTAL, 51};   // 0.5x
            case 201: return {SPEED_PORTAL, 102};  // 1x
            case 202: return {SPEED_PORTAL, 153};  // 2x
            case 203: return {SPEED_PORTAL, 204};  // 3x
            case 1334: return {SPEED_PORTAL, 255}; // 4x
            default: break;
        }
        switch (obj->m_objectType) {
            case GameObjectType::Hazard:
            case GameObjectType::AnimatedHazard: return {HAZARD, 255};
            case GameObjectType::Solid:
            case GameObjectType::Breakable:
            case GameObjectType::Slope: return {SOLID, 255};
            case GameObjectType::PinkJumpPad: return {JUMP_PAD, 85};
            case GameObjectType::YellowJumpPad: return {JUMP_PAD, 170};
            case GameObjectType::RedJumpPad: return {JUMP_PAD, 255};
            case GameObjectType::SpiderPad: return {GRAVITY_PAD, 128};
            case GameObjectType::GravityPad: return {GRAVITY_PAD, 255};
            case GameObjectType::DashRing: return {JUMP_ORB, 40};
            case GameObjectType::PinkJumpRing: return {JUMP_ORB, 85};
            case GameObjectType::YellowJumpRing: return {JUMP_ORB, 170};
            case GameObjectType::CustomRing: return {JUMP_ORB, 200};
            case GameObjectType::RedJumpRing: return {JUMP_ORB, 255};
            case GameObjectType::TeleportOrb: return {GRAVITY_ORB, 20};
            case GameObjectType::GravityDashRing: return {GRAVITY_ORB, 40};
            case GameObjectType::DropRing: return {GRAVITY_ORB, 85};
            case GameObjectType::SpiderOrb: return {GRAVITY_ORB, 128};
            case GameObjectType::GreenRing: return {GRAVITY_ORB, 170};
            case GameObjectType::GravityRing: return {GRAVITY_ORB, 255};
            case GameObjectType::NormalGravityPortal: return {GRAVITY_PORTAL, 85};
            case GameObjectType::GravityTogglePortal: return {GRAVITY_PORTAL, 170};
            case GameObjectType::InverseGravityPortal: return {GRAVITY_PORTAL, 255};
            case GameObjectType::CubePortal: return {MODE_PORTAL, 32};
            case GameObjectType::ShipPortal: return {MODE_PORTAL, 64};
            case GameObjectType::BallPortal: return {MODE_PORTAL, 96};
            case GameObjectType::UfoPortal: return {MODE_PORTAL, 128};
            case GameObjectType::WavePortal: return {MODE_PORTAL, 160};
            case GameObjectType::RobotPortal: return {MODE_PORTAL, 192};
            case GameObjectType::SpiderPortal: return {MODE_PORTAL, 224};
            case GameObjectType::SwingPortal: return {MODE_PORTAL, 255};
            case GameObjectType::SoloPortal: return {OTHER_PORTAL, 30};
            case GameObjectType::DualPortal: return {OTHER_PORTAL, 60};
            case GameObjectType::NormalMirrorPortal: return {OTHER_PORTAL, 90};
            case GameObjectType::InverseMirrorPortal: return {OTHER_PORTAL, 110};
            case GameObjectType::RegularSizePortal: return {OTHER_PORTAL, 128};
            case GameObjectType::TeleportPortal: return {OTHER_PORTAL, 200};
            case GameObjectType::MiniSizePortal: return {OTHER_PORTAL, 255};
            default: return {-1, 0};
        }
    }

    int gameMode(PlayerObject* p) {
        if (p->m_isShip) return 1;
        if (p->m_isBall) return 2;
        if (p->m_isBird) return 3;
        if (p->m_isDart) return 4;
        if (p->m_isRobot) return 5;
        if (p->m_isSpider) return 6;
        if (p->m_isSwing) return 7;
        return 0;
    }

    void buildCache(GJBaseGameLayer* layer) {
        s_objects.clear();
        for (auto obj : CCArrayExt<GameObject*>(layer->m_objects)) {
            auto look = lookFor(obj);
            if (look.channel >= 0) s_objects.push_back({obj, look});
        }
        std::sort(s_objects.begin(), s_objects.end(), [](auto const& a, auto const& b) {
            return a.obj->getPositionX() < b.obj->getPositionX();
        });
        s_cacheBuilt = true;
        log::info("GD AI: {} relevant objects cached", s_objects.size());
    }

    // Marks the objects around `player` into one grid (KINDS channels from `offset`).
    void observeGrid(PlayerObject* player, int offset, float cell) {
        auto pos = player->getPosition();
        bool flipped = player->m_isUpsideDown;
        float left = pos.x - cell;
        float bottom = pos.y - cell * ROWS / 2.f;
        float right = left + cell * COLS;
        float top = bottom + cell * ROWS;
        // Objects can be wider than a cell, so search a bit outside the grid.
        auto first = std::lower_bound(s_objects.begin(), s_objects.end(), left - 120.f,
            [](CachedObject const& o, float x) { return o.obj->getPositionX() < x; });
        for (auto it = first; it != s_objects.end() && it->obj->getPositionX() < right + 120.f; ++it) {
            auto const& rect = it->obj->getObjectRect();
            if (rect.getMaxX() < left || rect.getMinX() > right) continue;
            if (rect.getMaxY() < bottom || rect.getMinY() > top) continue;
            int c0 = std::max(0, static_cast<int>((rect.getMinX() - left) / cell));
            int c1 = std::min(COLS - 1, static_cast<int>((rect.getMaxX() - left) / cell));
            int r0 = std::max(0, static_cast<int>((rect.getMinY() - bottom) / cell));
            int r1 = std::min(ROWS - 1, static_cast<int>((rect.getMaxY() - bottom) / cell));
            for (int r = r0; r <= r1; r++) {
                // Upside down, flip rows so "down" is always the way gravity pulls.
                int row = flipped ? ROWS - 1 - r : r;
                for (int c = c0; c <= c1; c++) {
                    auto& value = s_grid[((offset + it->look.channel) * ROWS + row) * COLS + c];
                    value = std::max(value, it->look.value);
                }
            }
        }
    }

    // Writes one player's numbers (gravity, mode, speed...) into out[0..14].
    void observePlayer(PlayLayer* pl, PlayerObject* player, float* out) {
        auto pos = player->getPosition();
        bool flipped = player->m_isUpsideDown;
        int mode = gameMode(player);
        float yVel = static_cast<float>(player->m_yVelocity);
        out[0] = (flipped ? -yVel : yVel) / 20.f;
        out[1] = player->m_isOnGround ? 1.f : 0.f;
        out[2] = flipped ? 1.f : 0.f;
        out[3 + mode] = 1.f; // 3..10: cube, ship, ball, ufo, wave, robot, spider, swing
        out[11] = player->m_vehicleSize < 0.9f ? 1.f : 0.f;
        out[12] = player->m_playerSpeed / 1.6f;
        out[13] = (pos.y - 105.f) / 300.f;
        // Flying modes live between a floor and ceiling centred on the portal.
        bool bounded = mode == 1 || mode == 2 || mode == 3 || mode == 4 || mode == 6 || mode == 7;
        if (bounded) out[14] = (pl->m_gameState.m_portalY - pos.y) / 300.f * (flipped ? -1.f : 1.f);
    }

    bool isDual(PlayLayer* pl) {
        return pl->m_gameState.m_isDualMode && pl->m_player2;
    }

    // The grid only says which 30px cell something is in; this gives the exact distance, size,
    // kind and movement of the closest things ahead, so the AI can time clicks to the pixel.
    void observeNearest(PlayerObject* player, float* out) {
        auto pos = player->getPosition();
        float flip = player->m_isUpsideDown ? -1.f : 1.f;
        struct Near { float score; CachedObject const* o; CCRect rect; };
        std::vector<Near> found;
        auto first = std::lower_bound(s_objects.begin(), s_objects.end(), pos.x - 120.f,
            [](CachedObject const& o, float x) { return o.obj->getPositionX() < x; });
        for (auto it = first; it != s_objects.end() && it->obj->getPositionX() < pos.x + 600.f; ++it) {
            auto rect = it->obj->getObjectRect();
            if (rect.getMaxX() < pos.x - 15.f) continue; // already behind
            float dx = std::max(0.f, rect.getMinX() - pos.x);
            float dy = rect.getMidY() - pos.y;
            if (std::abs(dy) > 300.f) continue;
            found.push_back({dx + 0.7f * std::abs(dy), &*it, rect});
        }
        size_t n = std::min<size_t>(NEAREST, found.size());
        std::partial_sort(found.begin(), found.begin() + n, found.end(),
            [](Near const& a, Near const& b) { return a.score < b.score; });
        for (size_t i = 0; i < n; i++) {
            auto const& f = found[i];
            auto objPos = f.o->obj->getPosition();
            float* v = out + i * NEAREST_VALUES;
            v[0] = (f.rect.getMinX() - pos.x) / 300.f;
            v[1] = (f.rect.getMidY() - pos.y) / 150.f * flip;
            v[2] = f.rect.size.width / 60.f;
            v[3] = f.rect.size.height / 60.f;
            v[4] = (f.o->look.channel + 1) / 10.f; // 0.1 spike, 0.2 block, 0.3+ pads, orbs, portals
            if (auto last = s_lastSeen.find(f.o->obj); last != s_lastSeen.end()) {
                v[5] = (objPos.x - last->second.x) / DECISION_TICKS / 5.f;
                v[6] = (objPos.y - last->second.y) / DECISION_TICKS / 5.f * flip;
            }
            s_lastSeen[f.o->obj] = objPos;
        }
    }

    void observe(PlayLayer* pl) {
        std::fill(std::begin(s_grid), std::end(s_grid), 0);
        std::fill(std::begin(s_extra), std::end(s_extra), 0.f);
        observeGrid(pl->m_player1, NEAR_P1, CELL);
        observeGrid(pl->m_player1, FAR_P1, FAR_CELL);
        observePlayer(pl, pl->m_player1, s_extra);
        s_extra[15] = s_held ? 1.f : 0.f;
        if (isDual(pl)) {
            // Both players follow the same button, so the AI has to keep both alive.
            observeGrid(pl->m_player2, NEAR_P2, CELL);
            s_extra[16] = 1.f;
            observePlayer(pl, pl->m_player2, s_extra + 17); // 17..31
        }
        s_extra[32] = std::min(s_ticksSinceSwitch, 120) / 120.f;
        observeNearest(pl->m_player1, s_extra + 33);
    }
    void setButton(GJBaseGameLayer* layer, bool hold) {
        if (hold == s_held) return;
        layer->handleButton(hold, 1, true);
        s_held = hold;
        s_ticksSinceSwitch = 0;
    }

    float checkpointX(CheckpointObject* cp) {
        return cp->m_player1Checkpoint ? cp->m_player1Checkpoint->m_position.x : 0.f;
    }

    CheckpointObject* lastCheckpoint(PlayLayer* pl) {
        auto arr = pl->m_checkpointArray;
        if (!arr || arr->count() == 0) return nullptr;
        return static_cast<CheckpointObject*>(arr->lastObject());
    }

    void removeLastCheckpoint(PlayLayer* pl) {
        auto last = lastCheckpoint(pl);
        if (!last) return;
        s_checkpointMeta.erase(last);
        pl->removeCheckpoint(s_removeFirstArg);
        if (lastCheckpoint(pl) == last) {
            // Guessed the flag wrong and it popped the first one instead.
            s_removeFirstArg = !s_removeFirstArg;
            log::warn("GD AI: removeCheckpoint flag flipped to {}", s_removeFirstArg);
            pl->removeCheckpoint(s_removeFirstArg);
        }
    }

    void dropDoomedCheckpoints(PlayLayer* pl) {
        if (s_doomedAttempt < 0) return;
        while (auto last = lastCheckpoint(pl)) {
            auto it = s_checkpointMeta.find(last);
            if (it == s_checkpointMeta.end() || it->second.attempt != s_doomedAttempt) break;
            if (it->second.tick < s_deathTick - DOOMED_TICKS) break;
            removeLastCheckpoint(pl);
        }
        s_doomedAttempt = -1;
    }

    void maybePlaceCheckpoint(PlayLayer* pl) {
        // Modes that walk only get checkpoints on the ground, never mid-jump,
        // where the state might already be heading into a spike.
        auto steady = [](PlayerObject* p) {
            int mode = gameMode(p);
            bool walks = mode == 0 || mode == 2 || mode == 5 || mode == 6;
            return !walks || p->m_isOnGround;
        };
        if (!steady(pl->m_player1) || (isDual(pl) && !steady(pl->m_player2))) return;
        float x = pl->m_player1->getPositionX();
        // A GD checkpoint stores the whole level's state, which is slow on big levels. Practice only
        // ever starts at or before the coach's start, so checkpoints past it would just be thrown away.
        if (x > s_practiceStart + CHECKPOINT_SPACING) return;
        auto last = lastCheckpoint(pl);
        if (last && x < checkpointX(last) + CHECKPOINT_SPACING) return;
        if (!last && x < CHECKPOINT_SPACING) return;
        if (auto cp = pl->markCheckpoint()) s_checkpointMeta[cp] = {s_attempt, s_steps};
    }

    bool aiEnabled() {
        return Mod::get()->getSettingValue<bool>("enabled");
    }

    void saveReplay(PlayLayer* pl, float percent, bool won) {
        Replay r;
        r.levelID = pl->m_level->m_levelID.value();
        r.levelName = pl->m_level->m_levelName;
        r.percent = percent;
        r.won = won;
        r.ticks = s_steps;
        r.presses = s_presses;
        if (replay::saveIfBest(r) && won) {
            Notification::create("AI replay saved! Watch it from the GD AI panel", NotificationIcon::Success)->show();
        }
    }

    // Tells the brain which level is open and how observations are laid out.
    void sendHello(PlayLayer* pl) {
        std::string name = pl->m_level->m_levelName;
        Packet packet('H');
        packet.put(PROTOCOL).put(static_cast<int32_t>(pl->m_level->m_levelID.value()));
        packet.put(static_cast<int32_t>(name.size())).putBytes(name.data(), name.size());
        packet.put<int32_t>(CHANNELS).put<int32_t>(ROWS).put<int32_t>(COLS).put<int32_t>(EXTRA);
        packet.put(pl->m_levelLength);
        packet.send();
    }

    // Returns true on the frame the brain connects.
    bool pollBrain(PlayLayer* pl) {
        auto port = static_cast<uint16_t>(Mod::get()->getSettingValue<int64_t>("port"));
        if (!net::poll(port)) return false;
        sendHello(pl);
        log::info("GD AI: connected to brain on port {}", port);
        return true;
    }

    // Asks the brain how to start this attempt (from the start, practice from a
    // checkpoint, or switch to another level to review it), sets that up and
    // reports where it actually starts.
    bool startAttempt(PlayLayer* pl) {
        dropDoomedCheckpoints(pl);
        Packet ask('A');
        ask.put(pl->m_player1->getPositionX()).put<uint8_t>(pl->m_isPracticeMode ? 1 : 0);
        struct {
            uint8_t kind, flags;
            float targetX;
            int32_t levelID;
        } plan;
        if (!ask.send() || !net::recvAll(&plan.kind, 1) || !net::recvAll(&plan.flags, 1) ||
            !net::recvAll(&plan.targetX, 4) || !net::recvAll(&plan.levelID, 4)) return false;
        float targetX = plan.targetX;

        if (plan.kind == 2) {
            if (auto level = findLevel(plan.levelID)) {
                log::info("GD AI: brain switches to level {}", plan.levelID);
                s_switching = true;
                setButton(pl, false);
                switchToLevel(pl, level);
            } else {
                // GD doesn't have it (an online level never opened here); the brain picks something else.
                log::warn("GD AI: brain asked for level {}, which GD can't open", plan.levelID);
                Packet failed('F');
                failed.put<int32_t>(plan.levelID);
                failed.send();
            }
            return true;
        }
        s_quietWin = (plan.flags & 1) != 0;

        s_applyingPlan = true;
        setButton(pl, false);
        bool wantPractice = plan.kind == 1 && Mod::get()->getSettingValue<bool>("allow-practice");
        bool toggled = wantPractice != pl->m_isPracticeMode;
        s_practiceStart = wantPractice ? targetX : 0.f;
        if (toggled) {
            if (!wantPractice) s_checkpointMeta.clear();
            log::info("GD AI: coach turns practice mode {}", wantPractice ? "on" : "off");
            pl->togglePracticeMode(wantPractice);
        }
        if (wantPractice) {
            if (targetX <= 0.f) {
                while (lastCheckpoint(pl)) removeLastCheckpoint(pl);
            } else {
                while (auto last = lastCheckpoint(pl)) {
                    if (checkpointX(last) <= targetX) break;
                    removeLastCheckpoint(pl);
                }
            }
            pl->resetLevel(); // practice respawns at the last checkpoint left
            if (s_practiceStartsLogged < 20) {
                s_practiceStartsLogged++;
                auto cp = lastCheckpoint(pl);
                log::info("GD AI: practice start wanted <= {:.0f}, checkpoint at {:.0f}, player at {:.0f}, {} checkpoints",
                    targetX, cp ? checkpointX(cp) : 0.f, pl->m_player1->getPositionX(), pl->m_checkpointArray->count());
            }
        } else if (toggled) {
            pl->resetLevel();
        }
        s_applyingPlan = false;

        s_attempt++;
        s_windowTick = 0;
        s_lastSeen.clear();
        s_ticksSinceSwitch = 120;
        s_needPlan = false;
        // Only full runs from the start make replays; practice starts mid-level.
        s_presses.clear();
        s_recordingRun = !pl->m_isPracticeMode && pl->m_player1->getPositionX() < 5.f;
        auto cp = lastCheckpoint(pl);
        Packet begin('B');
        begin.put(pl->m_player1->getPositionX()).put<uint8_t>(pl->m_isPracticeMode ? 1 : 0);
        begin.put<uint8_t>(pl->m_isPracticeMode && cp ? 1 : 0);
        return begin.send();
    }

    void drawVision(PlayLayer* pl) {
        if (!s_visionNode) return;
        s_visionNode->clear();
        if (!Mod::get()->getSettingValue<bool>("show-ai-vision")) return;

        auto pos = pl->m_player1->getPosition();
        bool flipped = pl->m_player1->m_isUpsideDown;
        float left = pos.x - CELL;
        float bottom = pos.y - CELL * ROWS / 2.f;
        auto at = [](int channel, int row, int c) { return s_grid[(channel * ROWS + row) * COLS + c]; };
        for (int r = 0; r < ROWS; r++) {
            int row = flipped ? ROWS - 1 - r : r;
            for (int c = 0; c < COLS; c++) {
                ccColor4F fill = {1, 1, 1, 0.04f};
                if (at(HAZARD, row, c)) fill = {1, 0.15f, 0.15f, 0.45f};
                else if (at(SOLID, row, c)) fill = {0.2f, 1, 0.3f, 0.3f};
                else if (at(JUMP_PAD, row, c) || at(GRAVITY_PAD, row, c) || at(JUMP_ORB, row, c) || at(GRAVITY_ORB, row, c))
                    fill = {1, 0.9f, 0.1f, 0.4f};
                else if (at(GRAVITY_PORTAL, row, c) || at(MODE_PORTAL, row, c) || at(OTHER_PORTAL, row, c) || at(SPEED_PORTAL, row, c))
                    fill = {0.3f, 0.6f, 1, 0.3f};
                CCPoint verts[4] = {
                    {left + c * CELL, bottom + r * CELL},
                    {left + (c + 1) * CELL, bottom + r * CELL},
                    {left + (c + 1) * CELL, bottom + (r + 1) * CELL},
                    {left + c * CELL, bottom + (r + 1) * CELL},
                };
                s_visionNode->drawPolygon(verts, 4, fill, 0.5f, {1, 1, 1, 0.15f});
            }
        }
    }
}

class $modify(AIPlayLayer, PlayLayer) {
    bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects) {
        s_objects.clear();
        s_cacheBuilt = false;
        s_held = false;
        s_needPlan = true;
        s_applyingPlan = false;
        s_pendingReset = false;
        s_dying = false;
        s_won = false;
        s_inAIStep = false;
        s_steps = 0;
        s_doomedAttempt = -1;
        s_tickTime = 0.f;
        s_checkpointMeta.clear();
        std::fill(std::begin(s_grid), std::end(s_grid), 0);
        s_visionNode = nullptr;
        s_presses.clear();
        s_recordingRun = false;
        s_cursorShown = false;
        s_switching = false;
        s_quietWin = false;

        if (!PlayLayer::init(level, useReplay, dontCreateObjects)) return false;

        s_replay = replay::takeRequest(level->m_levelID.value());
        s_replayIndex = 0;
        s_replayFresh = s_replay.has_value();
        if (s_replay) {
            Notification::create(fmt::format("Playing the AI's {} ({:.0f}%)", s_replay->won ? "winning run" : "best run",
                s_replay->percent), NotificationIcon::Info)->show();
        }

        if (aiEnabled() && !m_isPlatformer) {
            // Afterflow hooks the same button function the AI presses, so the AI's clicks go through its buffer.
            if (auto afterflow = Loader::get()->getLoadedMod("afterflow.prsevend")) {
                log::info("GD AI: Afterflow {} is loaded; the AI's clicks go through its input buffer",
                    afterflow->getVersion().toVString());
            }
            if (net::connected()) sendHello(this);
            s_visionNode = CCDrawNode::create();
            s_visionNode->setZOrder(9999);
            m_objectLayer->addChild(s_visionNode);
        }
        return true;
    }

    void destroyPlayer(PlayerObject* player, GameObject* object) {
        if (s_replay && player == m_player1 && object != m_anticheatSpike) {
            float percent = this->getCurrentPercent();
            auto text = s_replay->won
                ? fmt::format("Replay went out of sync at {:.0f}%", percent)
                : fmt::format("The AI's best run ends here ({:.0f}%)", percent);
            log::info("GD AI: replay ended at {:.1f}% (saved run reached {:.1f}%)", percent, s_replay->percent);
            Notification::create(text, s_replay->won ? NotificationIcon::Warning : NotificationIcon::Info)->show();
            s_replay.reset();
            return PlayLayer::destroyPlayer(player, object);
        }
        // GD probes with the anti-cheat spike every tick; it never kills, and
        // swallowing it makes GD think noclip is on.
        bool ours = player == m_player1 || (player == m_player2 && isDual(this));
        if (!s_inAIStep || !ours || object == m_anticheatSpike) {
            return PlayLayer::destroyPlayer(player, object);
        }
        if (s_pendingReset) return;
        if (s_dying) return PlayLayer::destroyPlayer(player, object);

        auto pos = player->getPosition();
        if (player == m_player2 && s_player2DeathsLogged < 5) {
            s_player2DeathsLogged++;
            log::info("GD AI: player 2 died at {:.1f}% (dual mode)", this->getCurrentPercent());
        }
        Packet death('D');
        death.put(this->getCurrentPercent()).put(pos.x).put(pos.y);
        death.put<int32_t>(object ? object->m_objectID : -1);
        death.put<int32_t>(object ? static_cast<int32_t>(object->m_objectType) : -1);
        death.put<int32_t>(gameMode(player)).put<int32_t>(s_steps);
        death.put<uint8_t>(m_isPracticeMode ? 1 : 0);
        death.send();

        auto& record = stats::forLevel(m_level);
        record.attempts++;
        record.bestPercent = std::max(record.bestPercent, this->getCurrentPercent());
        stats::save();
        if (s_recordingRun && !m_isPracticeMode) saveReplay(this, this->getCurrentPercent(), false);

        s_doomedAttempt = s_attempt;
        s_deathTick = s_steps;
        setButton(this, false);
        if (Mod::get()->getSettingValue<bool>("normal-death")) {
            // Explode and restart exactly like a normal attempt.
            s_dying = true;
            return PlayLayer::destroyPlayer(player, object);
        }
        // Skip the death animation and the 1s delay, restart next frame.
        s_pendingReset = true;
    }

    void levelComplete() {
        if (s_replay) {
            s_replay.reset();
            s_won = true; // leave the end screen alone until a retry
            return PlayLayer::levelComplete();
        }
        if (!net::connected() || s_won) return PlayLayer::levelComplete();

        Packet win('W');
        win.put<int32_t>(s_steps).put<uint8_t>(m_isPracticeMode ? 1 : 0);
        win.send();
        setButton(this, false);
        if (m_isPracticeMode) {
            // Reaching the end in practice doesn't count; keep training.
            s_pendingReset = true;
            return;
        }

        auto& record = stats::forLevel(m_level);
        record.attempts++;
        record.bestPercent = 100.f;
        if (!record.completed) {
            record.completed = true;
            record.attemptsToComplete = record.attempts;
        }
        stats::save();
        (void)Mod::get()->saveData();
        if (s_recordingRun) saveReplay(this, 100.f, true);
        if (s_quietWin) {
            // Already beaten before (or a review run): count it and keep training.
            s_pendingReset = true;
            return;
        }
        s_won = true;
        PlayLayer::levelComplete();
    }

    void pauseGame(bool unfocused) {
        // The AI keeps playing when the window loses focus (alt-tab, a popup);
        // pausing with Esc still works.
        if (unfocused && (s_replay || (net::connected() && aiEnabled()))) return;
        PlayLayer::pauseGame(unfocused);
    }

    void resetLevel() {
        s_steps = 0;
        s_dying = false;
        s_tickTime = 0.f;
        if (!s_applyingPlan) s_needPlan = true;
        s_won = false; // retry from the end screen hands control back to the AI
        PlayLayer::resetLevel();
        s_cursorShown = false;
    }

    void resume() {
        PlayLayer::resume();
        s_cursorShown = false;
    }
};

class $modify(AIGameLayer, GJBaseGameLayer) {
    void update(float dt) {
        auto pl = PlayLayer::get();
        if (pl && static_cast<GJBaseGameLayer*>(pl) == this && aiEnabled() && (s_replay || net::connected())) {
            // You're only watching, so keep the mouse usable (pause button, menus). GD hides it in
            // more places than can be listed, so check every frame; the every-30-frames call is a
            // fallback in case the hidden flag can't be trusted. Showing it again is harmless.
            static int frames = 0;
            if (!s_cursorShown || CCEGLView::get()->m_bCursorHidden || ++frames % 30 == 0) {
                PlatformToolbox::showCursor();
                s_cursorShown = true;
            }
        }
        if (pl && static_cast<GJBaseGameLayer*>(pl) == this && s_replay) return playReplay(pl, dt);
        if (!pl || static_cast<GJBaseGameLayer*>(pl) != this || s_won || !aiEnabled() || pl->m_isPlatformer) {
            return GJBaseGameLayer::update(dt);
        }
        if (!net::connected()) {
            if (!pollBrain(pl)) return GJBaseGameLayer::update(dt);
            // The brain just connected mid-level: give it a clean attempt.
            s_pendingReset = true;
        }

        if (s_pendingReset) {
            s_pendingReset = false;
            setButton(this, false);
            pl->resetLevel();
            return;
        }
        // While the death animation plays, let GD run on its own until it restarts.
        if (s_dying) return GJBaseGameLayer::update(dt);
        if (s_switching) return;
        if (!s_cacheBuilt) buildCache(this);
        if (s_needPlan) {
            if (!startAttempt(pl)) log::warn("GD AI: lost connection to brain");
            return;
        }

        // Turn real time into whole 1/240s ticks so the game runs at the set
        // speed whatever the FPS.
        auto speed = static_cast<float>(Mod::get()->getSettingValue<double>("speed"));
        s_tickTime = std::min(s_tickTime + dt * speed, 0.25f);
        int steps = static_cast<int>(s_tickTime / TICK);
        s_tickTime -= steps * TICK;
        auto frameStart = std::chrono::steady_clock::now();
        for (int i = 0; i < steps; i++) {
            if (i > 0 && std::chrono::steady_clock::now() - frameStart > FRAME_BUDGET) {
                // Out of time this frame: the ticks left over run next frame (same ticks, same result).
                s_tickTime = std::min(s_tickTime + (steps - i) * TICK, 0.25f);
                break;
            }
            if (s_windowTick == 0) {
                // Start of a decision window: the brain answers with hold/release
                // (bit 2) and the tick in the window to do it on (bits 0-1).
                observe(pl);
                Packet state('S');
                state.put(pl->m_player1->getPositionX()).put(pl->getCurrentPercent());
                state.putBytes(s_extra, sizeof(s_extra)).putBytes(s_grid, sizeof(s_grid));
                uint8_t action = 0;
                if (!state.send() || !net::recvAll(&action, 1)) {
                    log::warn("GD AI: lost connection to brain");
                    setButton(this, false);
                    break;
                }
                s_planHold = (action & 4) != 0;
                s_planTick = action & 3;
                s_windowStartHeld = s_held;
            }
            bool hold = s_windowTick >= s_planTick ? s_planHold : s_windowStartHeld;
            if (s_recordingRun && hold != s_held) s_presses.push_back({s_steps, hold});
            setButton(this, hold);
            s_windowTick = (s_windowTick + 1) % DECISION_TICKS;
            s_ticksSinceSwitch++;

            s_inAIStep = true;
            GJBaseGameLayer::update(TICK);
            s_inAIStep = false;
            s_steps++;

            if (s_pendingReset || s_dying || s_won || s_needPlan) break;
            if (pl->m_isPracticeMode) maybePlaceCheckpoint(pl);
        }
        drawVision(pl);
    }

    // Plays the saved presses back at normal speed, no brain involved.
    void playReplay(PlayLayer* pl, float dt) {
        // Let GD finish opening the level first; resetting during the intro
        // would put the tick count out of step with the real start.
        if (!pl->m_started) return GJBaseGameLayer::update(dt);
        if (s_replayFresh) {
            // Start from a clean reset, exactly like the AI's recorded runs did.
            s_replayFresh = false;
            setButton(this, false);
            pl->resetLevel();
            return;
        }
        s_tickTime = std::min(s_tickTime + dt, 0.25f);
        int steps = static_cast<int>(s_tickTime / TICK);
        s_tickTime -= steps * TICK;
        for (int i = 0; i < steps && s_replay; i++) {
            auto const& presses = s_replay->presses;
            while (s_replayIndex < presses.size() && presses[s_replayIndex].first <= s_steps) {
                setButton(this, presses[s_replayIndex].second);
                s_replayIndex++;
            }
            GJBaseGameLayer::update(TICK);
            s_steps++;
        }
    }
};
