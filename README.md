# Алгоритм

1. Определение углов и сторон на фото для поиска листов бумаги, минимум два угла для положительного результата, учитывая поворот листа на фото
2. Определение расширений файлов: jpeg, jpg, png (std::filesystem::path::extension())
3. Итерация по папкам в поисках фото (std::filesystem::recursive_directory_iterator / std::filesystem::directory_iterator)
