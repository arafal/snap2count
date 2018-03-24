from django.conf.urls import url

from . import views

urlpatterns = [
	url('upload_pic/', views.upload_pic, name='upload_pic'),
    url('', views.index, name='index'),
    
]