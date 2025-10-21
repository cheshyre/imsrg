add_library(
    IMSRGBoostZip
    gzip.cpp
    zlib.cpp
)
target_link_libraries(
    IMSRGBoostZip
    Boost::boost
    # ZLIB::ZLIB
)