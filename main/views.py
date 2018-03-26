from django.shortcuts import render
from django.http import HttpResponse
from forms import ImageUploadForm
from models import ImageModel


def home(request):
    return render(request,'main/home.html')
    
def contact(request):
    return render(request,'main/contact.html')

def about(request):
    return render(request,'main/about.html')

def upload(request):
    return render(request,'main/upload.html')

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
