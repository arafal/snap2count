from django.conf.urls import url

from . import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
	url('about/', views.about, name='about'),
	url('contact/', views.contact, name='contact'),
	url('upload/', views.upload, name='upload'),
	url('upload_pic/', views.upload_pic, name='upload_pic'),
        url('home/', views.home, name='home'),
	url('', views.home, name='home'),
# ]
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) #for media folder
