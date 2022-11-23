from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.template import loader
from .forms import PhotoForm
from .models import Photo

def index(request):
    template = loader.get_template('firstapp/index.html')
    context = {'form':PhotoForm()}
    return HttpResponse(template.render(context,request))

def predict(request):
    if not request.method == 'POST':
        return
        redirect('firstapp:index')
    form = PhotoForm(request.POST,request.FILES)
    if not form.is_valid():
        return redirect('firstapp:index')
        raise ValueError('Formが不正です')

    photo = Photo(image=form.cleaned_data['image'])
    predicted, percentage = photo.predict()
    template = loader.get_template('firstapp/result.html')
    context = {'predicted':predicted,'percentage':percentage}
    return HttpResponse(template.render(context,request))