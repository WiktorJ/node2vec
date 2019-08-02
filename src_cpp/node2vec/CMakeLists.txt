add_executable(
        node2vec ${OBJS}
        biasedrandomwalk.cpp
        biasedrandomwalk.h
        n2v.cpp
        n2v.h
        node2vec.cpp
        stdafx.cpp
        stdafx.h
        targetver.h
        ${EXSNAPADV}/word2vec.cpp
)

