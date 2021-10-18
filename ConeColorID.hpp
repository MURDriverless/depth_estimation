#pragma once
#include <iostream>

enum class ConeColorID {
    Blue = 0,
    Orange = 1,
    Yellow = 2
};

inline std::string ConeColorID2str(ConeColorID coneColorID) {
    switch (coneColorID) {
        case (ConeColorID::Blue)    : return "BLUE";
        case (ConeColorID::Orange)  : return "ORANGE";
        case (ConeColorID::Yellow)  : return "YELLOW";
    }
}