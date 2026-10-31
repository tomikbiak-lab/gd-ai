#include <Geode/Geode.hpp>
#include <Geode/modify/MenuLayer.hpp>
#include "levels.hpp"
#include "replay.hpp"
#include "stats.hpp"

using namespace geode::prelude;

namespace {
    struct Line {
        std::string text;
        ccColor3B color;
        int64_t replayLevel = 0; // shows a REPLAY button when set
    };

    void openReplay(int64_t levelID) {
        auto level = findLevel(levelID);
        if (!level) {
            Notification::create("Open this level once first, then watch its replay", NotificationIcon::Warning)->show();
            return;
        }
        replay::request(levelID);
        openLevel(level);
    }
}

// Panel on the main menu: total attempts, hardest level beaten and every
// level the AI has finished (plus the ones it's still learning), with a
// button to watch the AI's saved run of each.
class $modify(AIMenuLayer, MenuLayer) {
    bool init() {
        if (!MenuLayer::init()) return false;

        std::vector<Line> lines;
        auto hardest = stats::hardestCompleted();
        lines.push_back({fmt::format("Total attempts: {}", stats::totalAttempts()), {255, 255, 255}});
        lines.push_back({hardest
            ? fmt::format("Hardest: {} ({})", hardest->name, hardest->difficultyName())
            : std::string("Hardest: nothing yet"), {255, 200, 60}});

        auto levels = stats::all();
        std::sort(levels.begin(), levels.end(), [](auto const& a, auto const& b) {
            if (a.completed != b.completed) return a.completed;
            return a.difficultyScore() > b.difficultyScore();
        });
        for (auto const& s : levels) {
            int64_t replayLevel = s.id != 0 && replay::exists(s.id) ? s.id : 0;
            if (s.completed) {
                lines.push_back({fmt::format("{} - {} - beaten in {} attempts", s.name, s.difficultyName(), s.attemptsToComplete),
                    {120, 255, 120}, replayLevel});
            } else if (s.attempts > 0) {
                lines.push_back({fmt::format("{} - learning, best {:.0f}% ({} attempts)", s.name, s.bestPercent, s.attempts),
                    {200, 200, 200}, replayLevel});
            }
        }
        constexpr size_t MAX_LINES = 10;
        if (lines.size() > MAX_LINES) {
            size_t hidden = lines.size() - (MAX_LINES - 1);
            lines.resize(MAX_LINES - 1);
            lines.push_back({fmt::format("... and {} more", hidden), {200, 200, 200}});
        }

        constexpr float WIDTH = 190.f;
        constexpr float LINE = 11.f;
        constexpr float BUTTON_W = 30.f;
        float height = 22.f + lines.size() * LINE + 6.f;

        auto panel = CCNode::create();
        panel->setID("gd-ai-hud"_spr);
        panel->setContentSize({WIDTH, height});
        panel->setAnchorPoint({0.f, 0.5f});

        auto bg = CCScale9Sprite::create("square02_001.png", {0, 0, 80, 80});
        bg->setContentSize({WIDTH, height});
        bg->setAnchorPoint({0, 0});
        bg->setOpacity(110);
        panel->addChild(bg);

        auto title = CCLabelBMFont::create("GD AI", "goldFont.fnt");
        title->setScale(0.55f);
        title->setPosition({WIDTH / 2.f, height - 11.f});
        panel->addChild(title);

        auto menu = CCMenu::create();
        menu->setPosition({0, 0});
        panel->addChild(menu);

        float y = height - 26.f;
        for (auto const& line : lines) {
            auto label = CCLabelBMFont::create(line.text.c_str(), "chatFont.fnt");
            label->setAnchorPoint({0.f, 0.5f});
            label->setColor(line.color);
            label->limitLabelWidth(WIDTH - 12.f - (line.replayLevel ? BUTTON_W : 0.f), 0.5f, 0.2f);
            label->setPosition({6.f, y});
            panel->addChild(label);
            if (line.replayLevel) {
                auto text = CCLabelBMFont::create("REPLAY", "goldFont.fnt");
                text->setScale(0.32f);
                auto button = CCMenuItemSpriteExtra::create(text, this, menu_selector(AIMenuLayer::onReplay));
                button->setTag(static_cast<int>(line.replayLevel));
                button->setPosition({WIDTH - 4.f - BUTTON_W / 2.f, y});
                menu->addChild(button);
            }
            y -= LINE;
        }

        auto winSize = CCDirector::get()->getWinSize();
        panel->setPosition({6.f, winSize.height / 2.f + 10.f});
        this->addChild(panel, 10);

        // For recording: jump straight into a main level once per game start,
        // optionally playing the AI's saved run of it.
        static bool autoplayDone = false;
        auto autoplayID = Mod::get()->getSettingValue<int64_t>("autoplay-level");
        if (autoplayID > 0 && !autoplayDone) {
            autoplayDone = true;
            bool watch = Mod::get()->getSettingValue<bool>("autoplay-replay");
            Loader::get()->queueInMainThread([autoplayID, watch] {
                if (watch && replay::exists(autoplayID)) return openReplay(autoplayID);
                openLevel(GameLevelManager::get()->getMainLevel(static_cast<int>(autoplayID), false));
            });
        }
        return true;
    }

    void onReplay(CCObject* sender) {
        openReplay(static_cast<CCNode*>(sender)->getTag());
    }
};
