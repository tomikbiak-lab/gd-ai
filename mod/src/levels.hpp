#pragma once
#include <Geode/Geode.hpp>

// Finds a level GD can open: main levels always, online ones only if they
// were opened once (so they're saved locally). Null when it isn't available.
inline GJGameLevel* findLevel(int64_t levelID) {
    auto glm = GameLevelManager::get();
    if (levelID <= 50) return glm->getMainLevel(static_cast<int>(levelID), false);
    return glm->getSavedLevel(static_cast<int>(levelID));
}

inline void openLevel(GJGameLevel* level) {
    cocos2d::CCDirector::get()->replaceScene(
        cocos2d::CCTransitionFade::create(0.5f, PlayLayer::scene(level, false, false)));
}

inline bool inTransition() {
    auto scene = cocos2d::CCDirector::get()->getRunningScene();
    return !scene || geode::cast::typeinfo_cast<cocos2d::CCTransitionScene*>(scene);
}

// Opens level once no level is running and no scene transition is in progress.
inline void openLevelWhenIdle(GJGameLevel* level, int frames = 0) {
    geode::Loader::get()->queueInMainThread([level, frames] {
        if (PlayLayer::get() || inTransition()) {
            if (frames > 0 && frames % 600 == 0) {
                geode::log::warn("GD AI: still waiting to open level {} ({} frames: {})", level->m_levelID.value(),
                    frames, PlayLayer::get() ? "the old level is still there" : "a scene transition is running");
            }
            return openLevelWhenIdle(level, frames + 1);
        }
        geode::log::info("GD AI: opening level {}", level->m_levelID.value());
        openLevel(level);
        level->release();
    });
}

// Leaves the current level the way the pause menu's exit does (GD's own clean-up,
// safe even in practice mode), then opens level. Swapping scenes straight from a
// running level can leave GD animations pointing at freed objects.
inline void leaveWhenIdle(PlayLayer* pl, GJGameLevel* level, int frames = 0) {
    // Next frame, outside the game update that asked for it, like a button press would.
    geode::Loader::get()->queueInMainThread([pl, level, frames] {
        // Leaving while GD is still fading into the level leaves it half-open: wait for the fade.
        if (inTransition()) return leaveWhenIdle(pl, level, frames + 1);
        geode::log::info("GD AI: leaving the level to switch (waited {} frames for a transition)", frames);
        pl->onQuit();
        pl->release();
        openLevelWhenIdle(level);
    });
}

inline void switchToLevel(PlayLayer* pl, GJGameLevel* level) {
    level->retain();
    pl->retain();
    leaveWhenIdle(pl, level);
}
