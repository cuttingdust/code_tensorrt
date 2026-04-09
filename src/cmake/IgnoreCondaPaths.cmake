# IgnoreCondaPaths.cmake
# 用法：include(IgnoreCondaPaths)

macro(ignore_conda_paths)
    if(MSVC)
        # 从环境变量中移除 Conda 路径
        set(conda_paths 
            "D:/MyApp/miniforge3"
            "C:/Users/31667/miniforge3"  # 可以加多个可能的路径
            "C:/ProgramData/miniforge3"
        )
        
        foreach(conda_root IN LISTS conda_paths)
            # 处理 INCLUDE
            if(DEFINED ENV{INCLUDE})
                string(REPLACE "${conda_root}/Library/include;" "" new_include "$ENV{INCLUDE}")
                string(REPLACE "${conda_root}/include;" "" new_include "${new_include}")
                set(ENV{INCLUDE} "${new_include}")
            endif()
            
            # 处理 LIB
            if(DEFINED ENV{LIB})
                string(REPLACE "${conda_root}/Library/lib;" "" new_lib "$ENV{LIB}")
                string(REPLACE "${conda_root}/lib;" "" new_lib "${new_lib}")
                set(ENV{LIB} "${new_lib}")
            endif()
            
            # 添加到 IGNORE_PATH
            list(APPEND CMAKE_IGNORE_PATH
                "${conda_root}/Library/include"
                "${conda_root}/Library/lib"
                "${conda_root}/include"
                "${conda_root}/lib"
            )
        endforeach()
        
        # 去重
        if(CMAKE_IGNORE_PATH)
            list(REMOVE_DUPLICATES CMAKE_IGNORE_PATH)
        endif()
        
        message(STATUS "已移除 Conda 路径：${conda_paths}")
    endif()
endmacro()