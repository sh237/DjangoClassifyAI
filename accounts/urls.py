from django.urls import path
from .views import AccountRegistration,Logout,Login,Welcome,top
from django.conf import settings
from django.conf.urls.static import static

app_name = 'accounts'
urlpatterns = [
    path('register', AccountRegistration.as_view(), name="register"),
    path("logout",Logout,name="logout"),
    path('login', Login,name='login'),
    path('top', top,name='top'),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

