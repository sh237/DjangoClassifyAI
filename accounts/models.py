from django.db import models
from django.contrib.auth.models import (BaseUserManager, AbstractBaseUser)
import uuid


class UserManager(BaseUserManager):
    def create_user(self, email, username, password=None):
        if not email:
            raise ValueError('Users must have an email address')

        user = self.model(
            email=self.normalize_email(email),
            username=username,
        )

        user.set_password(password)
        user.save(using=self._db)
        print(user.username)
        return user

    def create_superuser(self, email,username, password):
        user = self.create_user(
            email,
            password=password,
            username=username,
        )
        user.is_admin = True
        user.save(using=self._db)
        return user


class User(AbstractBaseUser):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(
        verbose_name='email',
        max_length=255,
        unique=True,
    )
    username = models.CharField( max_length=30, blank=False)
    is_active = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=False)
    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return self.email +" "+ str(self.id)

    def has_perm(self, perm, obj=None):
        return True

    def has_module_perms(self, app_label):
        return True

    @property
    def is_staff(self):
        return self.is_admin

CHOISES = (('0','スニーカー判別'),('1','説明文生成'))
class Photo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='user_photos')
    image = models.ImageField(upload_to='photos')
    uploaded_at = models.DateTimeField(auto_now_add=True, null=True)
    result_sentence = models.TextField(blank=True, null=True)
    category = models.CharField(choices=CHOISES, max_length=1, default='0')

    def __str__(self):
        return self.image.url
    
