cmake_minimum_required(VERSION 3.20)
project(test_cuda)

set(DPS_LIBRARYS
	${CUDA_MOUDLES}
)

cpp_execute(test_cuda)   # 确保 main.cu 在项目里

if(MSVC)
    #设为启动项

    # set_property(DIRECTORY ${CMAKE_BINARY_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME}) 
    set_target_properties(${PROJECT_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${OUT_RUN_PATH})
    set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "XArray")

    # set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "/GL")
    # set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "/LTCG")
	
	target_compile_options(${PROJECT_NAME} PRIVATE 
		$<$<COMPILE_LANGUAGE:CXX>:/utf-8>
	)
endif()