from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.template import loader
from .forms import PhotoForm
from .models import Photo
from accounts import models

def index(request):
    template = loader.get_template('image2text/index.html')
    context = {'form':PhotoForm()}
    return HttpResponse(template.render(context,request))

def predict(request):
    if not request.method == 'POST':
        return
        redirect('image2text:index')
    form = PhotoForm(request.POST,request.FILES)
    if not form.is_valid():
        return redirect('image2text:index')
        raise ValueError('Formが不正です')

    photo = Photo(image=form.cleaned_data['image'])
    english, japanese, graph, image = photo.predict()
    if request.user.is_authenticated:
        models.Photo.objects.create(image=form.cleaned_data['image'],user=request.user,result_sentence=japanese)
    template = loader.get_template('image2text/result.html')
    context = {'english':english,'japanese':japanese,'graph':graph,'image':image}
    return HttpResponse(template.render(context,request))