import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from chat_frontend.lm import query_neural_chat

@csrf_exempt
def get_response(request):
    global conversation_history
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('prompt', '').strip()

    #calling neural chat model
    re=query_neural_chat(user_message)
    return JsonResponse({'text':re })