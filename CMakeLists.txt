cmake_minimum_required(VERSION 3.10)
project(HogParallelProject)

# 1. Tự động tạo file compile_commands.json cho clangd (Fix lỗi đỏ Neovim)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 2. Cấu hình C++14 và các cờ tối ưu
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fopenmp")

# 3. Tìm các thư viện cần thiết
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

# 4. Khai báo đường dẫn file Header (.h)
include_directories(include)

# 5. Danh sách các file nguồn
# Đảm bảo tên file khớp với trong folder src/
set(PROJECT_SOURCES
    src/main.cpp
    src/hog_serial.cpp
    src/hog_openmp.cpp
    src/timing.cpp
)

# 6. Tạo file thực thi
add_executable(hog_cpu_test ${PROJECT_SOURCES})

# 7. Liên kết thư viện (Link Libraries)
# --- SỬA LỖI TẠI ĐÂY ---

# Link OpenMP (nếu tìm thấy)
if(OpenMP_CXX_FOUND)
    target_link_libraries(hog_cpu_test PUBLIC OpenMP::OpenMP_CXX)
endif()

# Link OpenCV (Thêm từ khóa PUBLIC để đồng bộ)
target_link_libraries(hog_cpu_test PUBLIC ${OpenCV_LIBS})
