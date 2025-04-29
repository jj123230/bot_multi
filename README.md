<h1> Multilingual chat bot </h1>
Download the whole bot_multi folder as a bot, bot_multi.py will be the main file to activate the bot

1. Install the env by conda env create -f folder_path\bot_multi\requirement.yml
2. pip install protobuf==3.20.2 paddlepaddle paddleocr datasets==2.21.0
3. Set your host_ip, port_ip, env and DB_info in setting_config.py
4. Set your db structure in config
5. Edit the folder path in test.txt, use folder_path/image_detect/image_detect.py folder_path to activate the image_detect bot

The API will respond with multilingual, and could connect to local ollama llm
