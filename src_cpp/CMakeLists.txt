cmake_minimum_required(VERSION 3.12)
project(node2vec)



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

add_subdirectory(node2vec_ms)
add_subdirectory(node2vec)

SET_SOURCE_FILES_PROPERTIES(
        ${OBJS}
        PROPERTIES
        EXTERNAL_OBJECT true
        GENERATED true
)
