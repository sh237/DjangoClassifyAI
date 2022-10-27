from unicodedata import name
from django.urls import path
from .views import index,predict
from django.conf import settings
from django.conf.urls.static import static

app_name = 'firstapp'

urlpatterns = [
    path('', index,name='index'),
    path('predict',predict,name='predict'),

]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)