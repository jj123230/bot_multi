<h1> Multilingual chat bot </h1>
Download the whole bot_multi folder as a bot, bot_multi.py will be the main file to activate the bot

1. Install the env by conda env create -f folder_path\bot_multi\requirement.yml
2. Set your host_ip, port_ip, env and DB_info in setting_config.py
3. Set your db structure in config
4. Edit the folder path in test.txt, use folder_path/bot_multi/bot_multi.py folder_path to activate the multilingual chat bot

The API will respond with multilingual, and could connect to local ollama llm

<h2> Main functions </h2>
http://host_ip/port_ip/updatefaq: refresh the threshold in the model <br>
http://host_ip/port_ip/updatemodel: train model <br>
http://host_ip/port_ip/load_model: load model in local folder <br>
<br>
http://host_ip/port_ip/dscbot: chat bot that answers faq, parameter: text= original text, lan= language <br>
http://host_ip/port_ip/chat: chat bot, parameter: text= original text, lan= language <br>
http://host_ip/port_ip/gen: generates synonyms sentences, parameter: text= original text, n= how many sentences <br>
http://host_ip/port_ip/translate: translate bot, parameter: text= original text, original_lang= original language, translate_lang= translate language <br>
http://host_ip/port_ip/judge_single: judge translate quality, parameter: original_text= = original text, translate_text= translate text, original_lang= original language, translate_lang= translate language <br>
http://host_ip/port_ip/judge_translate: judge translate quality for the whole dataset, parameter: env= dataset env, original_lang= original language, translate_lang= translate language <br>
