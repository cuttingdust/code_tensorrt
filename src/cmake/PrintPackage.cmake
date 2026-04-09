# PrintPackage.cmake
function(print_package package_name)
    message(">> ${package_name} ================================")
    
    # 基础信息
    if(DEFINED ${package_name}_FOUND)
        message("${package_name}_FOUND = ${${package_name}_FOUND}")
    endif()
    
    # 版本
    foreach(suffix VERSION VERSION_STRING VER)
        if(DEFINED ${package_name}_${suffix})
            message("${package_name}_${suffix} = ${${package_name}_${suffix}}")
            break()
        endif()
    endforeach()
    
    # 库文件 - 多种可能的命名
    set(possible_lib_suffixes 
        LIBRARIES LIBS LIBRARY LIB 
        LIBRARY_RELEASE LIBRARY_DEBUG
        IMPLIB_LINKER ITEMS)
    
    foreach(suffix ${possible_lib_suffixes})
        if(DEFINED ${package_name}_${suffix})
            message("${package_name}_${suffix} = ${${package_name}_${suffix}}")
        endif()
    endforeach()
    
    # 头文件目录
    set(possible_inc_suffixes 
        INCLUDE_DIRS INCLUDE_DIR 
        INCLUDES INCLUDE_PATH
        INCLUDE_DIRECTORIES)
    
    foreach(suffix ${possible_inc_suffixes})
        if(DEFINED ${package_name}_${suffix})
            message("${package_name}_${suffix} = ${${package_name}_${suffix}}")
        endif()
    endforeach()
    
    # 如果有目标，打印目标属性
    if(TARGET ${package_name}::${package_name})
        print_target_info(${package_name}::${package_name})
    endif()
endfunction()

function(print_target_info target_name)
    if(NOT TARGET ${target_name})
        return()
    endif()
    
    message("  Target: ${target_name}")
    
    get_target_property(inc_dirs ${target_name} INTERFACE_INCLUDE_DIRECTORIES)
    if(inc_dirs)
        message("    INTERFACE_INCLUDE_DIRECTORIES = ${inc_dirs}")
    endif()
    
    get_target_property(link_libs ${target_name} INTERFACE_LINK_LIBRARIES)
    if(link_libs)
        message("    INTERFACE_LINK_LIBRARIES = ${link_libs}")
    endif()
    
    get_target_property(import_loc ${target_name} IMPORTED_LOCATION)
    if(import_loc)
        message("    IMPORTED_LOCATION = ${import_loc}")
    endif()
endfunction()

# 带查找功能的封装
macro(find_package_and_print package_name)
    set(multi_value_args COMPONENTS TARGETS)
    cmake_parse_arguments(FPP "" "" "${multi_value_args}" ${ARGN})
    
    if(FPP_COMPONENTS)
        find_package(${package_name} REQUIRED COMPONENTS ${FPP_COMPONENTS})
    else()
        find_package(${package_name} REQUIRED)
    endif()
    
    print_package(${package_name})
    
    foreach(target ${FPP_TARGETS})
        print_target_info(${target})
    endforeach()
endmacro()