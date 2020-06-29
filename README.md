# SignDetection

## Подготовка окружения

    pip install -r requirements.txt

## Запуск приложения 
Так как размер весов предобученной сети больше, чем вес 
ограничение по размеру файла, то файл весов будет загружен АВТОМАТИЧЕСКИ
(это накладывает некоторые ограничения при запуске приложения, так как это занимает определенное время). 

Приложение можно запустить командой

    python app.py

После запуска приложения подождите, когда приложение напишет в консоль 
       
    all
    * erving Flask app "app" (lazy loading)
    * Environment: production
    WARNING: This is a development server. Do not use it in a production deployment.
    Use a production WSGI server instead.
    * Debug mode: off
    * Running on http://127.0.0.1:8080/ (Press CTRL+C to quit)
    
Теперь можно перейти по ссылке указанной в выводе приложения