from django.conf.urls import url

from . import views

urlpatterns = [
	url('about/', views.about, name='about'),
	url('upload/', views.upload, name='upload'),
	url('upload_pic/', views.upload_pic, name='upload_pic'),
        url('home/', views.home, name='home'),
	url('', views.home, name='home'),
    
]
