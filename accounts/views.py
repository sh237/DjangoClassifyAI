from django.shortcuts import render
from django.views.generic import TemplateView # テンプレートタグ
from .forms import AccountForm # ユーザーアカウントフォーム
from django.contrib.auth import authenticate
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.template import loader
from .models import Photo
class Welcome(TemplateView):
    template_name = "accounts/welcome.html"

class  AccountRegistration(TemplateView):

    def __init__(self):
        self.params = {
        "AccountCreate":False,
        "account_form": AccountForm(),
        }

    # Get処理
    def get(self,request):
        self.params["account_form"] = AccountForm()
        self.params["AccountCreate"] = False
        return render(request,"accounts/register.html",context=self.params)

    # Post処理
    def post(self,request):
        self.params["account_form"] = AccountForm(data=request.POST)

        # フォーム入力の有効検証
        if self.params["account_form"].is_valid():
            # アカウント情報をDB保存
            account = self.params["account_form"].save()
            # パスワードをハッシュ化
            account.set_password(account.password)
            # ハッシュ化パスワード更新
            account.save()

            # アカウント作成情報更新
            self.params["AccountCreate"] = True

        else:
            # フォームが有効でない場合
            print(self.params["account_form"].errors)

        return render(request,"accounts/register.html",context=self.params)

#ログイン
def Login(request):
    # POST
    if request.method == 'POST':
        # フォーム入力のユーザーID・パスワード取得
        Email = request.POST.get('email')
        Pass = request.POST.get('password')

        # Djangoの認証機能
        user = authenticate(username=Email, password=Pass)

        # ユーザー認証
        if user:
            #ユーザーアクティベート判定
            if user.is_active:
                # ログイン
                login(request,user)
                # ホームページ遷移
                return HttpResponseRedirect(reverse('accounts:top'))
            else:
                # アカウント利用不可
                return HttpResponse("アカウントが有効ではありません")
        # ユーザー認証失敗
        else:
            return HttpResponse("ログインIDまたはパスワードが間違っています")
    # GET
    else:
        return render(request, 'accounts/login.html')

def top(request):
    template = loader.get_template('accounts/top.html')
    if request.user.is_authenticated:
        if Photo.objects.filter(user__id=request.user.id).exists():
            context = {'photo':Photo.objects.filter(user__id=request.user.id).order_by('-id')}
        else:
            context = {'user':request.user}
    return HttpResponse(template.render(context,request))

@login_required
def Logout(request):
    logout(request)
    # ログイン画面遷移
    return HttpResponseRedirect(reverse('accounts:login'))

