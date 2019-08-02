cmake_minimum_required(VERSION 3.12)
project(src_cpp)

set(CMAKE_CXX_STANDARD 17)
set(SNAPROOT /Users/w.jurasz/Studies/GR/snap)
set(GLIB  glib-core)
set(SNAP  snap-core)
set(GLIBADV  glib-adv)
set(SNAPADV  snap-adv)
set(SOURCEROOT node2vec_ms)


set(EXGLIB ${SNAPROOT}/${GLIB})
set(EXSNAP ${SNAPROOT}/${SNAP})
set(EXGLIBADV ${SNAPROOT}/${GLIBADV})
set(EXSNAPADV ${SNAPROOT}/${SNAPADV})

set(OBJS ${EXSNAP}/Snap.o )

include_directories(${SOURCEROOT} ${EXGLIBADV} ${EXGLIB} ${EXSNAP} ${EXSNAPADV})

add_executable(node2vec ${OBJS}
        ${SOURCEROOT}/biasedrandomwalk.cpp
        ${SOURCEROOT}/biasedrandomwalk.h
        ${SOURCEROOT}/n2v.cpp
        ${SOURCEROOT}/n2v.h
        ${SOURCEROOT}/node2vec.cpp
        ${SOURCEROOT}/stdafx.cpp
        ${SOURCEROOT}/stdafx.h
        ${SOURCEROOT}/targetver.h
        ${EXSNAPADV}/word2vec.cpp
)

SET_SOURCE_FILES_PROPERTIES(
        ${OBJS}
        PROPERTIES
        EXTERNAL_OBJECT true
        GENERATED true
)

