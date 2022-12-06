from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.template import loader
from .forms import TextForm
from .models import Text
from accounts import models

def index(request):
    template = loader.get_template('summarization/index.html')
    context = {'form':TextForm()}
    return HttpResponse(template.render(context,request))

def predict(request):
    if not request.method == 'POST':
        return
    form = TextForm(request.POST)
    if not form.is_valid():
        return redirect('summarization:index')
    

    text = Text(text=form.cleaned_data['text'])
    print(text)
    original, summary  = text.predict()
    # if request.user.is_authenticated:
    #     models.Photo.objects.create(image=form.cleaned_data['image'],user=request.user,result_sentence=japanese)
    template = loader.get_template('summarization/result.html')
    context = {'original':original,'summary':summary}
    return HttpResponse(template.render(context,request))