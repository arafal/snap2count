from django.shortcuts import render
from django.http import HttpResponse
from forms import ImageUploadForm
from models import ImageModel


def index(request):
    return render(request,'main/home.html')
# Create your views here.

def upload_pic(request):
	if request.method == 'POST':
		form = ImageUploadForm(request.POST , request.FILES)
		if form.is_valid():
			m = ImageModel()
			m.model = form.cleaned_data['image']
			data = form.cleaned_data
			m.save()
			return render(request,'main/success.html',{'data':data})
	return HttpResponseForbidden('allowed only via POST')